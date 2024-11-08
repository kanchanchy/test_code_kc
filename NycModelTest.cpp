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
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/ml_functions/functions.h"
//#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/Embedding.h"
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
#include <chrono>
#include <locale>
#include "velox/functions/Udf.h"
#include <unordered_map>
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;




class CPUUtilizationTracker {
public:
    CPUUtilizationTracker() {
        startCPUTime = getCPUTime();
        startWallTime = std::chrono::steady_clock::now();
    }

    ~CPUUtilizationTracker() {
        auto endWallTime = std::chrono::steady_clock::now();
        double endCPUTime = getCPUTime();

        // Calculate total CPU time (user + system)
        double cpuTimeUsed = endCPUTime - startCPUTime;

        // Calculate real elapsed time (wall time)
        auto elapsedWallTime = std::chrono::duration_cast<std::chrono::milliseconds>(endWallTime - startWallTime).count() / 1000.0;

        // CPU utilization as a percentage
        double cpuUtilization = (cpuTimeUsed / elapsedWallTime) * 100.0;

        std::cout << "CPU Utilization for the method: " << cpuUtilization << "%\n";
    }

private:
    double startCPUTime;
    std::chrono::steady_clock::time_point startWallTime;

    // Method to get current CPU time (user + system)
    double getCPUTime() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        double userCPUTime = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
        double sysCPUTime = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
        return userCPUTime + sysCPUTime;
    }
};




class VectorAddition : public MLFunction {
 public:
  VectorAddition(int inputDims) {
    inputDims_ = inputDims;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = rightHolder->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input1Matrix(input1Values, numInput, inputDims_);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input2Matrix(input2Values, numInput, inputDims_);

    std::vector<std::vector<float>> results;

    for (int i = 0; i < numInput; i++) {
      //Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> vSum = input1Matrix.row(i) + input2Matrix.row(i);
      Eigen::VectorXf vSum = input1Matrix.row(i) + input2Matrix.row(i);
      std::vector<float> curVec(vSum.data(), vSum.data() + vSum.size());
      //std::vector<float> std_vector(vSum.data(), vSum.data() + vSum.size());
      results.push_back(curVec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "vector_addition";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int inputDims_;
};



class GetOrderFeatures : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    const char* dateFormat = "%Y-%m-%d";
    int secondsInADay = 86400;
    std::string days_of_week[7] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    //auto cAddressNums = decodedArray0->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();
    //auto cCustFlags = decodedArray1->base()->as<FlatVector<int>>();

    for (int i = 0; i < rows.size(); i++) {
        std::vector<float> vec;

        StringView valTimestamp = decodedArray0->valueAt<StringView>(i);
        std::string sTimestamp = std::string(valTimestamp);
        StringView valWeekday = decodedArray1->valueAt<StringView>(i);
        std::string sWeekday = std::string(valWeekday);

        struct std::tm t = {};
        std::istringstream ss(sTimestamp);
        ss >> std::get_time(&t, dateFormat);

      // Check if parsing was successful
      if (ss.fail()) {
          std::cerr << "Failed to parse date string " << sTimestamp << std::endl;
          vec.push_back(0.0);
          continue;
      }
      else {
          // Convert tm struct to time_t (timestamp)
          time_t tt = mktime(&t);
          // Cast time_t to int64_t
          int64_t longTimestamp = static_cast<int64_t>(tt);
          vec.push_back((static_cast<float>(longTimestamp/secondsInADay))/15340.0);
      }

      for (int i = 0; i < 7; i++) {
          if (sWeekday == days_of_week[i]) {
              vec.push_back(1.0);
          }
          else {
              vec.push_back(0.0);
          }
      }
      results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    //auto localResult = maker.flatVector<int64_t>(results);
    //context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .argumentType("VARCHAR")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_order_features";
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



class IsPopularStore : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto* flatResult = output->asFlatVector<int32_t>();
    BaseVector* parameter = args[0].get();

    exec::LocalDecodedVector vecHolder(context, *parameter, rows);
    auto decodedArray = vecHolder.get();
    auto arrayVector = decodedArray->base()->as<ArrayVector>();
    auto flatElements = arrayVector->elements()->asFlatVector<float>();
    //auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>()->elements();
    //float* input1Values = baseLeftArray->values()->asMutable<float>();

    /*std::vector<int> results;

    for (int i = 0; i < rows.size(); i++) {
      //std::vector<float> vecStore = decodedArray->valueAt<std::vector<float>>(i);
      //float sumRes = std::accumulate(vecStore.begin(), vecStore.end(), 0.0);
      //float meanRes = sumRes / vecStore.size();

      auto arrayVector = decodedArray->base()->as<ArrayVector>();
      auto arrayIndex = decodedArray->index(i);
      auto size = arrayVector->sizeAt(arrayIndex);
      auto offset = arrayVector->offsetAt(arrayIndex);

      float sumRes = 0.0;
      for (int j = 0; j < size; ++j) {
        float element = arrayVector->elements()->asFlatVector<float>()->valueAt(offset + j);
        sumRes += element;
      }
      float meanRes = sumRes / size;

      if (meanRes >= 0.5) {
          results.push_back(1);
      }
      else {
          results.push_back(0);
      }
    } */

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->set(row, 0);  // Handle nulls
        return;
      }

      auto arrayIndex = decodedArray->index(row);
      int size = arrayVector->sizeAt(arrayIndex);  // Get the size of the array
      int offset = arrayVector->offsetAt(arrayIndex);  // Get the offset

      // Use std::accumulate directly over the flat elements vector
      float sum = std::accumulate(
          flatElements->rawValues() + offset,
          flatElements->rawValues() + offset + size,
          0.0f);
      float firstElem = flatElements->valueAt(offset);
      float mean = (sum - firstElem) / (size - 1);
      flatResult->set(row, mean >= 0.4f ? 1 : 0);  // Directly set result
    });

    //VectorMaker maker{context.pool()};
    //auto localResult = maker.flatVector<int>(results);
    //context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "is_popular_store";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class GetMaxIndex : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    auto* flatResult = output->asFlatVector<int32_t>();

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto arrayVector = decodedArray->base()->as<ArrayVector>();
    auto flatElements = arrayVector->elements()->asFlatVector<float>();
    //auto inputProbs = decodedArray->base()->as<ArrayVector>();
    //auto inputProbsValues = inputProbs->elements()->asFlatVector<float>();

    /*std::vector<int> results;
    for (int i = 0; i < rows.size(); i++) {
        std::vector<float> inputProbs = decodedArray->valueAt<std::vector<float>>(i);
        auto max_iter = std::max_element(inputProbs.begin(), inputProbs.end());
        int index_of_max = std::distance(inputProbs.begin(), max_iter);
        results.push_back(index_of_max);
    }

    VectorMaker maker{context.pool()};
    auto localResult = maker.flatVector<int>(results);
    context.moveOrCopyResult(localResult, rows, output);*/

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->setNull(row, true);  // Handle nulls
        return;
      }

      // Get the array index, size, and offset for this row
      auto arrayIndex = decodedArray->index(row);
      auto size = arrayVector->sizeAt(arrayIndex);
      auto offset = arrayVector->offsetAt(arrayIndex);

      // Find the index of the maximum element in the array
      auto maxIter = std::max_element(
          flatElements->rawValues() + offset,
          flatElements->rawValues() + offset + size);

      int index_of_max = std::distance(flatElements->rawValues() + offset, maxIter);
      flatResult->set(row, index_of_max);  // Set result in output vector
    });

  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("ARRAY(REAL)")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "get_max_index";
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


class GetDistance : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<double> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();
    BaseVector* base2 = args[2].get();
    BaseVector* base3 = args[3].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();

    exec::LocalDecodedVector thirdHolder(context, *base2, rows);
    auto decodedArray2 = thirdHolder.get();

    exec::LocalDecodedVector fourthHolder(context, *base3, rows);
    auto decodedArray3 = fourthHolder.get();

    for (int i = 0; i < rows.size(); i++) {

        double lat1 = decodedArray0->valueAt<double>(i);
        double lon1 = decodedArray1->valueAt<double>(i);
        double lat2 = decodedArray2->valueAt<double>(i);
        double lon2 = decodedArray3->valueAt<double>(i);

        double distance = std::sqrt(std::pow(lat1 - lat2, 2) + std::pow(lon1 - lon2, 2));
        results.push_back(distance);
    }

    VectorMaker maker{context.pool()};
    //output = maker.arrayVector<float>(results, REAL());
    auto localResult = maker.flatVector<double>(results);
    context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("DOUBLE")
                .argumentType("DOUBLE")
                .argumentType("DOUBLE")
                .argumentType("DOUBLE")
                .returnType("DOUBLE")
                .build()};
  }

  static std::string getName() {
    return "get_distance";
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



class NycModelTest : public HiveConnectorTestBase {
 public:
  NycModelTest() {
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

  ~NycModelTest() {}

  void registerFunctions(std::string modelFilePath, int numCols);
  void registerNNFunctions(int numCols, int leftCols, int splitNeuron, std::string modelPath);
  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath);

  RowVectorPtr getCensusData(std::string filePath);
  RowVectorPtr getEarningData(std::string filePath);
  std::vector<std::vector<float>> loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint);
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

void NycModelTest::registerFunctions(std::string modelFilePath, int numCols) {

  exec::registerVectorFunction(
          "get_order_features",
          GetOrderFeatures::signatures(),
          std::make_unique<GetOrderFeatures>());
  std::cout << "Completed registering function for get_order_features" << std::endl;

  exec::registerVectorFunction(
          "is_popular_store",
          IsPopularStore::signatures(),
          std::make_unique<IsPopularStore>());
  std::cout << "Completed registering function for is_popular_store" << std::endl;

  exec::registerVectorFunction(
            "get_max_index",
            GetMaxIndex::signatures(),
            std::make_unique<GetMaxIndex>());
  std::cout << "Completed registering function for get_binary_class" << std::endl;

  exec::registerVectorFunction(
          "get_distance",
          GetDistance::signatures(),
          std::make_unique<GetDistance>());
  std::cout << "Completed registering function for get_distance" << std::endl;

}


void NycModelTest::registerNNFunctions(int numCols, int leftCols, int splitNeuron, std::string modelPath) {
  std::vector<std::vector<float>> w1 = loadHDF5Array(modelPath, "fc1.weight", 0);
  std::vector<std::vector<float>> b1 = loadHDF5Array(modelPath, "fc1.bias", 0);
  std::vector<std::vector<float>> w2 = loadHDF5Array(modelPath, "fc_extras.0.weight", 0);
  std::vector<std::vector<float>> b2 = loadHDF5Array(modelPath, "fc_extras.0.bias", 0);
  std::vector<std::vector<float>> w11 = loadHDF5Array(modelPath, "w11", 0);
  std::vector<std::vector<float>> w12 = loadHDF5Array(modelPath, "w12", 0);

  auto itemNNweight1Vector = maker.arrayVector<float>(w1, REAL());
  auto itemNNweight2Vector = maker.arrayVector<float>(w2, REAL());
  auto itemNNBias1Vector = maker.arrayVector<float>(b1, REAL());
  auto itemNNBias2Vector = maker.arrayVector<float>(b2, REAL());
  auto itemNNweight11Vector = maker.arrayVector<float>(w11, REAL());
  auto itemNNweight12Vector = maker.arrayVector<float>(w12, REAL());


  exec::registerVectorFunction(
      "convert_int_array",
      ConvertToIntArray::signatures(),
      std::make_unique<ConvertToIntArray>());

  exec::registerVectorFunction(
      "mat_mul_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight1Vector->elements()->values()->asMutable<float>()),
          numCols,
          splitNeuron));

  exec::registerVectorFunction(
      "mat_vector_add_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias1Vector->elements()->values()->asMutable<float>()), splitNeuron));

  exec::registerVectorFunction(
      "mat_mul_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight2Vector->elements()->values()->asMutable<float>()),
          splitNeuron,
          2));

  exec::registerVectorFunction(
      "mat_vector_add_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias2Vector->elements()->values()->asMutable<float>()), 2));

  exec::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(),
          {},
          true);

  exec::registerVectorFunction(
      "softmax", Softmax::signatures(), std::make_unique<Softmax>());

  exec::registerVectorFunction(
      "mat_mul_11",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight11Vector->elements()->values()->asMutable<float>()),
          leftCols,
          splitNeuron));

  exec::registerVectorFunction(
      "mat_mul_12",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight12Vector->elements()->values()->asMutable<float>()),
          numCols - leftCols,
          splitNeuron));

  exec::registerVectorFunction(
          "vector_addition",
          VectorAddition::signatures(),
          std::make_unique<VectorAddition>(splitNeuron));


}



ArrayVectorPtr NycModelTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

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

RowVectorPtr NycModelTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
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


std::vector<std::vector<float>> NycModelTest::loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint) {
    /*if (!std::filesystem::exists(filename)) {
          throw std::runtime_error("File not found: " + filename);
    }*/

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();

    // Get the number of dimensions
    int rank = dataspace.getSimpleExtentNdims();
    // std::cout << "Rank: " << rank << std::endl;

    // Allocate space for the dimensions
    std::vector<hsize_t> dims(rank);

    // Get the dataset dimensions
    dataspace.getSimpleExtentDims(dims.data(), nullptr);

    size_t rows;
    size_t cols;

    if (rank == 1) {
      rows = dims[0];
      cols = 1;
    } else if (rank == 2) {
      rows = dims[0];
      cols = dims[1];
    } else {
      throw std::runtime_error("Unsupported rank: " + std::to_string(rank));
    }

    //std::cout << "Num Rows: " << rows << ", Num Columns " << cols << std::endl;

    // Read data into a 1D vector
    std::vector<float> flatData(rows * cols);
    dataset.read(flatData.data(), H5::PredType::NATIVE_FLOAT);

    // Convert to 2D vector
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = flatData[i * cols + j];
            if (doPrint == 1)
                std::cout << result[i][j] << ", ";
        }
        if (doPrint == 1)
            std::cout << std::endl;
    }

    // Close the dataset and file
    dataset.close();
    file.close();

    return result;
}


RowVectorPtr NycModelTest::getCensusData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> cId;
    std::vector<double> cLat;
    std::vector<double> cLon;
    std::vector<int> cBucketLat;
    std::vector<int> cBucketLon;
    std::vector<std::vector<float>> cFeatures;


    std::string line;
    int latIndex;
    int lonIndex;
    int idIndex;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
        std::cout << "Ignoring header: " << line << std::endl;
        std::istringstream iss(line);
        std::string numberStr;
        int colIndex = 0;
        while (std::getline(iss, numberStr, ',')) {
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (numberStr == "lat_centroid") {
                latIndex = colIndex;
            }
            else if (numberStr == "lon_centroid") {
                lonIndex = colIndex;
            }
            else if (numberStr == "id_centroid") {
                idIndex = colIndex;
            }
            colIndex ++;
        }
    }

    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);

        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;
	    std::vector<float> features;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == latIndex) {
                double numTemp = std::stod(numberStr);
                cLat.push_back(numTemp);
                cBucketLat.push_back(static_cast<int>(numTemp * 100.0));
            }
            else if (colIndex == lonIndex) {
                double numTemp = std::stod(numberStr);
                cLon.push_back(numTemp);
                cBucketLon.push_back(static_cast<int>(numTemp * 100.0));
            }
            else if (colIndex == idIndex) {
                cId.push_back(std::stoi(numberStr));
            }
            else {
                features.push_back(std::stof(numberStr));
            }

	        colIndex ++;

        }
        cFeatures.push_back(features);

    }

    file.close();

     // Prepare Customer table
     auto cIdVector = maker.flatVector<int>(cId);
     auto cLatVector = maker.flatVector<double>(cLat);
     auto cLonVector = maker.flatVector<double>(cLon);
     auto cBucketLatVector = maker.flatVector<int>(cBucketLat);
     auto cBucketLonVector = maker.flatVector<int>(cBucketLon);
     auto cFeaturesVector = maker.arrayVector<float>(cFeatures, REAL());
     auto censusRowVector = maker.rowVector(
         {"c_id", "c_lat", "c_lon", "c_bucket_lat", "c_bucket_lon", "c_features"},
         {cIdVector, cLatVector, cLonVector, cBucketLatVector, cBucketLonVector, cFeaturesVector}
     );

     return censusRowVector;
}


RowVectorPtr NycModelTest::getEarningData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> eId;
    std::vector<int> eClass;
    std::vector<double> eLat;
    std::vector<double> eLon;
    std::vector<int> eBucketLat;
    std::vector<int> eBucketLon;
    std::vector<std::vector<float>> eFeatures;


    std::string line;
    int latIndex;
    int lonIndex;
    int idIndex;
    int classIndex;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
        std::cout << "Ignoring header: " << line << std::endl;
        std::istringstream iss(line);
        std::string numberStr;
        int colIndex = 0;
        while (std::getline(iss, numberStr, ',')) {
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (numberStr == "lat_intpt") {
                latIndex = colIndex;
            }
            else if (numberStr == "lon_intpt") {
                lonIndex = colIndex;
            }
            else if (numberStr == "id_intpt") {
                idIndex = colIndex;
            }
            else if (numberStr == "category") {
                classIndex = colIndex;
            }
            colIndex ++;
        }
    }

    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);

        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;
	    std::vector<float> features;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == latIndex) {
                double numTemp = std::stod(numberStr);
                eLat.push_back(numTemp);
                eBucketLat.push_back(static_cast<int>(numTemp * 100.0));
            }
            else if (colIndex == lonIndex) {
                double numTemp = std::stod(numberStr);
                eLon.push_back(numTemp);
                eBucketLon.push_back(static_cast<int>(numTemp * 100.0));
            }
            else if (colIndex == idIndex) {
                eId.push_back(std::stoi(numberStr));
            }
            else if (colIndex == classIndex) {
                eClass.push_back(std::stoi(numberStr));
            }
            else {
                features.push_back(std::stof(numberStr));
            }

	        colIndex ++;

        }
        eFeatures.push_back(features);

    }

    file.close();

     // Prepare Customer table
     auto eIdVector = maker.flatVector<int>(eId);
     auto eClassVector = maker.flatVector<int>(eClass);
     auto eLatVector = maker.flatVector<double>(eLat);
     auto eLonVector = maker.flatVector<double>(eLon);
     auto eBucketLatVector = maker.flatVector<int>(eBucketLat);
     auto eBucketLonVector = maker.flatVector<int>(eBucketLon);
     auto eFeaturesVector = maker.arrayVector<float>(eFeatures, REAL());
     auto earningRowVector = maker.rowVector(
         {"e_id", "e_class", "e_lat", "e_lon", "e_bucket_lat", "e_bucket_lon", "e_features"},
         {eIdVector, eClassVector, eLatVector, eLonVector, eBucketLatVector, eBucketLonVector, eFeaturesVector}
     );

     return earningRowVector;
}



void NycModelTest::testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string orderFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();
                      
     std::string path = dataFile->path;

     RowVectorPtr censusRowVector = getCensusData("resources/data/nyc_2000Census.csv");
     std::cout << "censusRowVector data generated" << std::endl;
     RowVectorPtr earningRowVector = getEarningData("resources/data/nyc_earning.csv");
     std::cout << "earningRowVector data generated" << std::endl;

     int totalRowsCensus = censusRowVector->size();
     int totalRowsEarning = earningRowVector->size();

     std::cout << "census data size: " << totalRowsCensus << ",  earning data size: " << totalRowsEarning << std::endl;

     int batch_counts = 4;
     int batchSizeCensus = totalRowsCensus / batch_counts;
     int batchSizeEarning = totalRowsEarning / batch_counts;

     std::vector<RowVectorPtr> batchesCensus;
     std::vector<RowVectorPtr> batchesEarning;

     for (int i = 0; i < batch_counts; ++i) {
         int start = i * batchSizeCensus;
         int end = (i == (batch_counts - 1)) ? totalRowsCensus : (i + 1) * batchSizeCensus;  // Handle remainder for last batch
         batchesCensus.push_back(std::dynamic_pointer_cast<RowVector>(censusRowVector->slice(start, end - start)));

         start = i * batchSizeEarning;
         end = (i == (batch_counts - 1)) ? totalRowsEarning : (i + 1) * batchSizeEarning;  // Handle remainder for last batch
         batchesEarning.push_back(std::dynamic_pointer_cast<RowVector>(earningRowVector->slice(start, end - start)));
     }

     registerNNFunctions(103, 56, 32, "resources/model/model_nyc_2_32_2.h5");
     CPUUtilizationTracker tracker;

     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();


    auto myPlan2 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({earningRowVector})
                         .localPartition({"e_bucket_lat", "e_bucket_lon"})
                         .project({"e_id", "e_lat", "e_lon", "e_class", "e_bucket_lat", "e_bucket_lon", "mat_mul_11(e_features) as dnn_part1"})
                         .hashJoin({"e_bucket_lat", "e_bucket_lon"},
                             {"c_bucket_lat", "c_bucket_lon"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({censusRowVector})
                             .localPartition({"c_bucket_lat", "c_bucket_lon"})
                             .project({"c_id", "c_lat", "c_lon", "c_bucket_lat", "c_bucket_lon", "mat_vector_add_1(mat_mul_12(c_features)) as dnn_part2"})
                             .planNode(),
                             "get_distance(e_lat, e_lon, c_lat, c_lon) <= 0.3",
                             {"e_id", "c_id", "e_class", "dnn_part1", "dnn_part2"}
                         )
                         .project({"e_id", "c_id", "e_class", "get_max_index(softmax(mat_vector_add_2(mat_mul_2(vector_addition(dnn_part1, dnn_part2))))) AS predicted_class"})
                         .planNode();


    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    auto results2 = exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << "Results Size: " << results2->size() << std::endl;
    std::cout << results2->toString(0, 10) << std::endl;
    std::cout << "Time for Executing with Single Batch (sec): " << std::endl;
    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count()) /1000.0 << std::endl;


 
}


void NycModelTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath) {

  std::cout << "Option is " << option << std::endl;
  registerFunctions(modelFilePath, 9);

  if (option == 0) {
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
DEFINE_int32(numCols, 9, "number of columns in the dataset to be predicted");
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

  NycModelTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}, orderDataFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath, orderDataFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath, orderDataFilePath);

}
