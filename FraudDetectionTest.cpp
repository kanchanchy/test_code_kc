/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <folly/init/Init.h>
#include <pstl/pstl_config.h>
#include <torch/torch.h>
#include <cwchar>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/XGBoost.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include <ctime>
#include <iomanip>
#include <time.h>
#include <locale>

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;



class IsWeekday : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<int> results;
    auto inputTimes = args[0]->as<FlatVector<int64_t>>();
    const int secondsInADay = 86400;
    for (int i = 0; i < rows.size(); i++) {
        int64_t timestamp = inputTimes->valueAt(i);
        // Calculate the number of days since Unix epoch
        int64_t daysSinceEpoch = timestamp / secondsInADay;

        // Unix epoch (Jan 1, 1970) was a Thursday, so dayOfWeek for epoch is 4 (0=Sunday, 6=Saturday)
        int dayOfWeekEpoch = 4;  // Thursday

        // Calculate the current day of the week (0=Sunday, ..., 6=Saturday)
        int dayOfWeek = (daysSinceEpoch + dayOfWeekEpoch) % 7;

        // Return true if the day is Saturday (6) or Sunday (0)
        if (dayOfWeek == 0 || dayOfWeek == 6) {
            results.push_back(0);
        }
        else {
            results.push_back(1);
        }
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "is_weekday";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class TimeDiffInDays : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    if (args.size() < 2 || !args[0] || !args[1]) {
        // Handle error
        std::cerr << "Two arguments for time diff not found" << std::endl;
        return;
    }

    std::vector<int64_t> results;
    auto inputTimes1 = args[0]->as<FlatVector<int64_t>>();
    auto inputTimes2 = args[1]->as<FlatVector<int64_t>>();
    int secondsInADay = 86400;
    for (int i = 0; i < rows.size(); i++) {
        int64_t timestamp1 = inputTimes1->valueAt(i);
        int64_t timestamp2 = inputTimes2->valueAt(i);

        int64_t differenceInSeconds = std::abs(timestamp1 - timestamp2);
        int64_t differenceInDays = differenceInSeconds / secondsInADay;

        results.push_back(differenceInDays);
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int64_t>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .argumentType("BIGINT")
                .returnType("BIGINT")
                .build()};
  }

  static std::string getName() {
    return "time_diff_in_days";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class DateToTimestamp : public MLFunction {
 public:
 DateToTimestamp (const char* dateFormat_) {
     dateFormat = dateFormat_;
 }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto inputStrings = args[0]->as<FlatVector<StringView>>();

    std::vector<int64_t> results;

    for (int i = 0; i < rows.size(); i++) {
      std::string inputStr = std::string(inputStrings->valueAt(i));// + " 00:00:00";

      struct std::tm t;
      std::istringstream ss(inputStr);
      ss >> std::get_time(&t, dateFormat);

      // Check if parsing was successful
      if (ss.fail()) {
          std::cerr << "Failed to parse date string " << inputStr << std::endl;
          results.push_back(0);
          continue;
      }

      // Convert tm struct to time_t (timestamp)
      time_t tt = mktime(&t);
      // Cast time_t to int64_t
      int64_t timestamp = static_cast<int64_t>(tt);
      results.push_back(timestamp);

    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int64_t>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("BIGINT")
                .build()};
  }

  static std::string getName() {
    return "date_to_timestamp";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

  private:
    const char* dateFormat;

};



class FraudDetectionTest : public HiveConnectorTestBase {
 public:
  FraudDetectionTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~FraudDetectionTest() {}

  void registerFunctions(std::string modelFilePath="resources/model/fraud_xgboost_1600_8", int numCols = 28);
  void registerNNFunctions(int numCols);
  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath);
  
  RowVectorPtr getCustomerData(int numCustomers, int numCustomerFeatures);
  RowVectorPtr getTransactionData(int numTransactions, int numTransactionFeatures, int numCustomers);
  RowVectorPtr getOrderData(std::string filePath);
  RowVectorPtr getTransactionData(std::string filePath);
  
  void testingNestedLoopJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingNestedLoopJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingHashJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingHashJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingHashJoinWithPredictFilter(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingHashJoinWithNeuralNetwork(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);

  ArrayVectorPtr parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols);

  RowVectorPtr writeDataToFile(std::string csvFilePath, int numRows, int numCols, int numDataSplits, 
                               std::string outPath, int dataBatchSize);


  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {
  }


  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};
  VectorMaker maker{pool_.get()};
};

void FraudDetectionTest::registerFunctions(std::string modelFilePath, int numCols) {

  std::cout <<"To register function for TreePrediction" << std::endl;

  exec::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(0, "resources/model/fraud_xgboost_10_8/0.txt", 28, false));

  std::cout << "To register function for XGBoost Prediction" << std::endl;

  exec::registerVectorFunction(
      "xgboost_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>(modelFilePath, numCols, true));

  std::cout << "To register function for Concatenation" << std::endl;

  exec::registerVectorFunction(
      "concat_vectors",
      Concat::signatures(),
      std::make_unique<Concat>(10, 18));
  
  exec::registerVectorFunction(
      "is_weekday",
      IsWeekday::signatures(),
      std::make_unique<IsWeekday>());
  std::cout << "Completed registering function for is_weekday" << std::endl;

  exec::registerVectorFunction(
        "date_to_timestamp_1",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%d"));
  std::cout << "Completed registering function for date_to_timestamp_1" << std::endl;

  exec::registerVectorFunction(
        "date_to_timestamp_2",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%dT%H:%M"));
  std::cout << "Completed registering function for date_to_timestamp_2" << std::endl;

  exec::registerVectorFunction(
        "time_diff_in_days",
        TimeDiffInDays::signatures(),
        std::make_unique<TimeDiffInDays>());
  std::cout << "Completed registering function for time_diff_in_days" << std::endl;

}


void FraudDetectionTest::registerNNFunctions(int numCols) {

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  randomGenerator.setFloatRange(-1, 1);

  std::vector<std::vector<float>> itemNNweight1 =
      randomGenerator.genFloat2dVector(numCols, 32);
  auto itemNNweight1Vector = maker.arrayVector<float>(itemNNweight1, REAL());

  std::vector<std::vector<float>> itemNNBias1 =
      randomGenerator.genFloat2dVector(32, 1);
  auto itemNNBias1Vector = maker.arrayVector<float>(itemNNBias1, REAL());

  std::vector<std::vector<float>> itemNNweight2 =
      randomGenerator.genFloat2dVector(32, 16);
  auto itemNNweight2Vector = maker.arrayVector<float>(itemNNweight2, REAL());

  std::vector<std::vector<float>> itemNNBias2 =
      randomGenerator.genFloat2dVector(16, 1);
  auto itemNNBias2Vector = maker.arrayVector<float>(itemNNBias2, REAL());

  std::vector<std::vector<float>> itemNNweight3 =
      randomGenerator.genFloat2dVector(16, 2);
  auto itemNNweight3Vector = maker.arrayVector<float>(itemNNweight3, REAL());

  std::vector<std::vector<float>> itemNNBias3 =
      randomGenerator.genFloat2dVector(2, 1);
  auto itemNNBias3Vector = maker.arrayVector<float>(itemNNBias3, REAL());

  exec::registerVectorFunction(
      "mat_mul_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight1Vector->elements()->values()->asMutable<float>(),
          numCols,
          32));

  exec::registerVectorFunction(
      "mat_vector_add_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          itemNNBias1Vector->elements()->values()->asMutable<float>(), 32));

  exec::registerVectorFunction(
      "mat_mul_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight2Vector->elements()->values()->asMutable<float>(),
          32,
          16));

  exec::registerVectorFunction(
      "mat_vector_add_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          itemNNBias2Vector->elements()->values()->asMutable<float>(), 16));

  exec::registerVectorFunction(
      "mat_mul_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight3Vector->elements()->values()->asMutable<float>(),
          16,
          2));

  exec::registerVectorFunction(
      "mat_vector_add_3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          itemNNBias3Vector->elements()->values()->asMutable<float>(), 2));

  exec::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>());

  exec::registerVectorFunction(
      "softmax", Softmax::signatures(), std::make_unique<Softmax>());

}


ArrayVectorPtr FraudDetectionTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

    int size = numRows * numCols;

    std::cout << "Loading tensor of size " << size << " from " << filePath << std::endl;

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<std::vector<float>> inputArrayVector;

    
    int index = 0;
    
    std::string line;
    
    while (numRows--) { // Read a line from the file

        std::vector<float> curRow(numCols);
	
        std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
						    //
            float number = std::stof(numberStr);    // Convert the string to float

	        if (colIndex < numCols)					    

                curRow[colIndex] = number;

	        colIndex ++;

        }

	    inputArrayVector.push_back(curRow);
    }

    file.close();

    ArrayVectorPtr tensor = maker.arrayVector<float>(inputArrayVector);
    
    return tensor;

}

RowVectorPtr FraudDetectionTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
                                                 int numDataSplits, std::string outPath, int dataBatchSize) {

    ArrayVectorPtr inputArrayVector = parseCSVFile(maker, csvFilePath, numRows, numCols);
  
    std::vector<int32_t> indexVector;
  
    for (int i = 0; i < numRows; i++) {
  
       indexVector.push_back(i);
  
    }
  
    auto inputIndexVector = maker.flatVector<int32_t>(indexVector);
  
    auto inputRowVector = maker.rowVector({"row_id", "x"}, {inputIndexVector, inputArrayVector});

    auto dataConfig = std::make_shared<facebook::velox::dwrf::Config>();
  
    // affects the number of splits
    // number of bites in each stripe (collection of rows)
    // strip size should be <= split size (total_size / total splits)
    // to have the desired number of splits
    uint64_t kDataSizeKB = 512UL;

    uint32_t numDataRows = dataBatchSize;
  
    dataConfig->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 799 * kDataSizeKB);
  
    dataConfig->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, numDataRows);
  
    // auto dataFile = TempFilePath::create();

    // outPath = dataFile->path;

    writeToFile(outPath, {inputRowVector}, dataConfig);

    return inputRowVector;

}


RowVectorPtr FraudDetectionTest::getOrderData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> oOrderId;
    std::vector<int> oCustomerSk;
    std::vector<std::string> oWeekday;
    std::vector<std::string> oDate;

    
    int index = 0;
    
    std::string line;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
        std::cout << "Ignoring header: " << line << std::endl;
    }
    
    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);
	
        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == 0) {
                oOrderId.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 1) {
                oCustomerSk.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 2) {
                oWeekday.push_back(numberStr);
            }
            else if (colIndex == 3) {
                oDate.push_back(numberStr);
            }

	        colIndex ++;

        }

	    //inputArrayVector.push_back(curRow);
        index ++;
        if (index == 3000) {
            break;
        }
    }

    file.close();

     // Prepare Customer table
     auto oOrderIdVector = maker.flatVector<int>(oOrderId);
     auto oCustomerSkVector = maker.flatVector<int>(oCustomerSk);
     auto oWeekdayVector = maker.flatVector<std::string>(oWeekday);
     auto oDateVector = maker.flatVector<std::string>(oDate);
     auto orderRowVector = maker.rowVector(
         {"o_order_id", "o_customer_sk", "o_weekday", "o_date"},
         {oOrderIdVector, oCustomerSkVector, oWeekdayVector, oDateVector}
     );

     return orderRowVector;
}


RowVectorPtr FraudDetectionTest::getTransactionData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<float> tAmount;
    std::vector<int> tSender;
    std::vector<std::string> tReceiver;
    std::vector<int64_t> transactionId;
    std::vector<std::string> tTime;


    int index = 0;

    std::string line;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
        std::cout << "Ignoring header: " << line << std::endl;
    }

    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);

        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == 0) {
                tAmount.push_back(std::stof(numberStr));
            }
            else if (colIndex == 1) {
                tSender.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 2) {
                tReceiver.push_back(numberStr);
            }
            else if (colIndex == 3) {
                transactionId.push_back(std::stoll(numberStr));
            }
            else if (colIndex == 4) {
                tTime.push_back(numberStr);
           }

	        colIndex ++;

        }

	    //inputArrayVector.push_back(curRow);
        index ++;
        if (index == 10000) {
            break;
        }
    }

    file.close();

     // Prepare Customer table
     auto tAmountVector = maker.flatVector<float>(tAmount);
     auto tSenderVector = maker.flatVector<int>(tSender);
     auto tReceiverVector = maker.flatVector<std::string>(tReceiver);
     auto transactionIdVector = maker.flatVector<int64_t>(transactionId);
     auto tTimeVector = maker.flatVector<std::string>(tTime);
     auto transactionRowVector = maker.rowVector(
         {"t_amount", "t_sender", "t_receiver", "transaction_id", "t_time"},
         {tAmountVector, tSenderVector, tReceiverVector, transactionIdVector, tTimeVector}
     );

     return transactionRowVector;
}


RowVectorPtr FraudDetectionTest::getCustomerData(int numCustomers, int numCustomerFeatures) {
    
    // Customer table
     std::vector<int64_t> customerIDs;
     std::vector<std::vector<float>> customerFeatures;
     
     // Populate Customer table
     for (int i = 0; i < numCustomers; ++i) {
         customerIDs.push_back(i);  // Example: Customer IDs from 0 to 499

         std::vector<float> features;
         for(int j=0; j < numCustomerFeatures; j++){
             features.push_back(2.0);
         }
         
         customerFeatures.push_back(features);
     }

     // Prepare Customer table
     auto customerIDVector = maker.flatVector<int64_t>(customerIDs);
     auto customerFeaturesVector = maker.arrayVector<float>(customerFeatures, REAL());
     auto customerRowVector = maker.rowVector(
         {"customer_id", "customer_features"},
         {customerIDVector, customerFeaturesVector}
     );

     return customerRowVector;
}


RowVectorPtr FraudDetectionTest::getTransactionData(int numTransactions, int numTransactionFeatures, int numCustomers) {
    
    // Transaction table
     std::vector<int64_t> transactionIDs;
     std::vector<int64_t> transactionCustomerIDs;
     std::vector<std::vector<float>> transactionFeatures;
     
     // Populate Transaction table
     for (int i = 0; i < numTransactions; ++i) {
         transactionIDs.push_back(i);  // Example: Transaction IDs from 0 to 4999
         
         // Randomly assign each transaction to a customer
         transactionCustomerIDs.push_back(rand() % numCustomers);

         std::vector<float> features;
         for(int j=0; j < numTransactionFeatures; j++){
             features.push_back(5.0);
         }
         
         transactionFeatures.push_back(features);
     }

     // Prepare Transaction table
     auto transactionIDVector = maker.flatVector<int64_t>(transactionIDs);
     auto transactionCustomerIDVector = maker.flatVector<int64_t>(transactionCustomerIDs);
     auto transactionFeaturesVector = maker.arrayVector<float>(transactionFeatures, REAL());
     auto transactionRowVector = maker.rowVector(
         {"transaction_id", "trans_customer_id", "transaction_features"},
         {transactionIDVector, transactionCustomerIDVector, transactionFeaturesVector}
     );

     return transactionRowVector;
}


void FraudDetectionTest::testingNestedLoopJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;

     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

     
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .filter("customer_id > 50")
                         .nestedLoopJoin(exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(), {"transaction_id", "trans_customer_id", "transaction_features", "customer_id", "customer_features"}
                         )
                         .filter("customer_id = trans_customer_id")
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Nested Noop Join and Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingNestedLoopJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

     
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .nestedLoopJoin(exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(), {"transaction_id", "trans_customer_id", "transaction_features", "customer_id", "customer_features"}
                         )
                         .filter("customer_id = trans_customer_id")
                         .filter("customer_id > 50")
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Nested Noop Join and without Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingHashJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .filter("customer_id > 50")
                         .hashJoin({"customer_id"},
                         {"trans_customer_id"},
                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(),
                         "",
                         {"customer_id", "customer_features", "transaction_id", "transaction_features"})
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Has Join and Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingHashJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .hashJoin({"customer_id"},
                         {"trans_customer_id"},
                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(),
                         "",
                         {"customer_id", "customer_features", "transaction_id", "transaction_features"})
                         .filter("customer_id > 50")
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Has Join and without Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingHashJoinWithPredictFilter(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .filter("customer_id > 50")
                         .hashJoin({"customer_id"},
                         {"trans_customer_id"},
                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(),
                         "",
                         {"customer_id", "customer_features", "transaction_id", "transaction_features"})
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .filter("decision_tree_predict(features) > 0.5")
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Has Join and Predit Filter (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingHashJoinWithNeuralNetwork(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     numCols = numCustomerFeatures + numTransactionFeatures;

     registerNNFunctions(numCustomerFeatures + numTransactionFeatures);
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);
     /*
     RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
     randomGenerator.setFloatRange(-1, 1);

     std::vector<std::vector<float>> itemNNweight1 = randomGenerator.genFloat2dVector(numCols, 32);
     auto itemNNweight1Vector = maker.arrayVector<float>(itemNNweight1, REAL());
     
     std::vector<std::vector<float>> itemNNBias1 = randomGenerator.genFloat2dVector(32, 1);
     auto itemNNBias1Vector = maker.arrayVector<float>(itemNNBias1, REAL());
     
     std::vector<std::vector<float>> itemNNweight2 = randomGenerator.genFloat2dVector(32, 16);
     auto itemNNweight2Vector = maker.arrayVector<float>(itemNNweight2, REAL());
     
     std::vector<std::vector<float>> itemNNBias2 = randomGenerator.genFloat2dVector(16, 1);
     auto itemNNBias2Vector = maker.arrayVector<float>(itemNNBias2, REAL());
     
     std::vector<std::vector<float>> itemNNweight3 = randomGenerator.genFloat2dVector(16, 2);
     auto itemNNweight3Vector = maker.arrayVector<float>(itemNNweight3, REAL());
     
     std::vector<std::vector<float>> itemNNBias3 = randomGenerator.genFloat2dVector(2, 1);
     auto itemNNBias3Vector = maker.arrayVector<float>(itemNNBias3, REAL());

     std::string compute =  NNBuilder()
                            .denseLayer(32, numCols,
                            itemNNweight1Vector->elements()->values()->asMutable<float>(), 
                            itemNNBias1Vector->elements()->values()->asMutable<float>(),
                            NNBuilder::RELU)
                            .denseLayer(16, 32,
                            itemNNweight2Vector->elements()->values()->asMutable<float>(), 
                            itemNNBias2Vector->elements()->values()->asMutable<float>(),
                            NNBuilder::RELU)
                            .denseLayer(2, 16,
                            itemNNweight3Vector->elements()->values()->asMutable<float>(), 
                            itemNNBias3Vector->elements()->values()->asMutable<float>(),
                            NNBuilder::SOFTMAX)
                            .build();*/

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .filter("customer_id > 50")
                         .hashJoin({"customer_id"},
                         {"trans_customer_id"},
                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(),
                         "",
                         {"customer_id", "customer_features", "transaction_id", "transaction_features"})
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .filter("xgboost_predict(features) > 0.5")
                         .project({"tid", "softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(mat_vector_add_1(mat_mul_1(features))))))))) AS label"})
                         //.project({"tid", fmt::format(compute, "features") + " AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Has Join and Neural Network (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string orderFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     RowVectorPtr orderRowVector = getOrderData(orderFilePath);
     std::cout << "orderRowVector data generated" << std::endl;
     RowVectorPtr transactionRowVector = getTransactionData("resources/data/financial_transactions.csv");
     std::cout << "transactionRowVector data generated" << std::endl;

     /*
     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     */
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({orderRowVector})
                         .project({"o_customer_sk", "o_order_id", "date_to_timestamp_1(o_date) AS o_timestamp"})
                         .filter("o_timestamp IS NOT NULL")
                         .filter("is_weekday(o_timestamp) = 1")
                         //.singleAggregation({"o_customer_sk"}, {"count(o_order_id) as total_order", "max(o_timestamp) as o_last_order_time"})
                         .singleAggregation({"o_customer_sk"}, {"max(o_timestamp) as o_last_order_time"})
                         .hashJoin({"o_customer_sk"},
                             {"t_sender"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({transactionRowVector})
                             .project({"t_amount", "t_sender", "t_receiver", "transaction_id", "date_to_timestamp_2(t_time) as t_timestamp"})
                             .filter("t_timestamp IS NOT NULL")
                             .planNode(),
                             "",
                             {"o_customer_sk", "o_last_order_time", "transaction_id", "t_amount", "t_timestamp"}
                         )
                         /*.filter("time_diff_in_days(o_last_order_time, t_timestamp) <= 7")
                         .project({"o_customer_sk", "transaction_id", "get_transaction_features(total_order, t_amount, t_timestamp) as transaction_features"})
                         .filter("is_anomalous(transaction_features) < 0.5")
                         .hashJoin({"o_customer_sk"},
                             {"c_customer_sk"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({customerRowVector})
                             .project({"c_customer_sk", "c_current_address_sk", "get_country(c_birth_country) as c_country", "get_age(c_birth_day, c_birth_month, c_birth_year) as c_age"})
                             .planNode(),
                             "",
                             {"transaction_id", "transaction_features", "c_current_address_sk", "c_country", "c_age"}
                         )
                         .project({"transaction_id", "get_all_features(transaction_features, c_current_address_sk, c_country, c_age) as all_features"})
                         .filter("xgboost_model(all_features) >= 0.5")
                         .project({"transaction_id", "dnn_model(all_features) as is_fraud"})*/
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << results->toString(0, 5) << std::endl;
   
    std::cout << "Time for Executing with Real Data (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath) {

  std::cout << "Option is " << option << std::endl;
  registerFunctions(modelFilePath, numCols);

  if (option == 0) {
      //testingNestedLoopJoinWithPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      //testingNestedLoopJoinWithoutPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      //testingHashJoinWithPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      //testingHashJoinWithoutPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      //testingHashJoinWithPredictFilter(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      //testingHashJoinWithNeuralNetwork(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      testingWithRealData(numDataSplits, dataBatchSize, numRows, numCols, orderDataFilePath, modelFilePath);
  }

  else
      std::cout << "Seleceted test is not supported" << std::endl;
}



DEFINE_int32(rewriteOrNot, 0, "0 for all tests");
DEFINE_int32(numDataSplits, 16, "number of data splits");
DEFINE_int32(numTreeSplits, 16, "number of tree splits");
DEFINE_int32(numTreeRows, 100, "batch size for processing trees");
DEFINE_int32(dataBatchSize, 100, "batch size for processing input samples");
DEFINE_int32(numRows, 10, "number of tuples in the dataset to be predicted");
DEFINE_int32(numCols, 10, "number of columns in the dataset to be predicted");
DEFINE_string(dataFilePath, "resources/data/creditcard_test.csv", "path to input dataset to be predicted");
DEFINE_string(modelFilePath, "resources/model/fraud_xgboost_1600_8", "path to the model used for prediction");
DEFINE_string(orderDataFilePath, "resources/data/order.csv", "path to input order dataset");

int main(int argc, char** argv) {

  setlocale(LC_TIME, "C");

  folly::init(&argc, &argv, false);

  memory::MemoryManager::initialize({});

  int option = FLAGS_rewriteOrNot;
  int numDataSplits = FLAGS_numDataSplits;
  int numTreeSplits = FLAGS_numTreeSplits;
  int numTreeRows = FLAGS_numTreeRows;
  int dataBatchSize = FLAGS_dataBatchSize;
  int numRows = FLAGS_numRows;
  int numCols = FLAGS_numCols;
  std::string dataFilePath = FLAGS_dataFilePath;
  std::string modelFilePath = FLAGS_modelFilePath;
  std::string orderDataFilePath = FLAGS_orderDataFilePath;

  FraudDetectionTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}, orderDataFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath, orderDataFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath, orderDataFilePath);

}
