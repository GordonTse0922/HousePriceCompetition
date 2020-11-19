using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Resourcer;

async Task<string> GetTrainDataFilePath()
{
    var tmpFilePath = Path.GetTempFileName();
    await using var fs = File.OpenWrite(tmpFilePath);
    await Resource.AsStream("train.csv").CopyToAsync(fs);
    return tmpFilePath;
}

async Task<string> GetTestDataFilePath()
{
    var tmpFilePath = Path.GetTempFileName();
    await using var fs = File.OpenWrite(tmpFilePath);
    await Resource.AsStream("test.csv").CopyToAsync(fs);
    return tmpFilePath;
}

static void CancelExperimentAfterAnyKeyPress(CancellationTokenSource cts)
{
    Task.Run(() =>
    {
        Console.WriteLine("Press any key to stop the experiment run...");
        Console.ReadKey();
        cts.Cancel();
    });
}

var ctx = new MLContext();
var (trainDataPath, testDataPath) = (await GetTrainDataFilePath(), await GetTestDataFilePath());

ConsoleHelper.ConsoleWriteHeader("=============== Inferring columns in dataset ===============");
var inferenceResults = ctx.Auto().InferColumns(await GetTrainDataFilePath(), "SalePrice", groupColumns: false);
ConsoleHelper.Print(inferenceResults);

var columnInfo = inferenceResults.ColumnInformation;
new List<String> { "Id", "SalePrice" }.ForEach(x => columnInfo.NumericColumnNames.Remove(x));
// new List<String> { "SalePrice" }.ForEach(columnInfo.IgnoredColumnNames.Add);

var textLoader = ctx.Data.CreateTextLoader(inferenceResults.TextLoaderOptions);
var trainDataView = textLoader.Load(trainDataPath);
var testDataView = textLoader.Load(testDataPath);

ConsoleHelper.ShowDataViewInConsole(ctx, trainDataView);

var cts = new CancellationTokenSource();
CancelExperimentAfterAnyKeyPress(cts);
var experimentSettings = new RegressionExperimentSettings()
{
    MaxExperimentTimeInSeconds = 3600,
    CancellationToken = cts.Token,
    OptimizingMetric = RegressionMetric.RootMeanSquaredError,
    CacheDirectory = null,
};

new List<RegressionTrainer> { RegressionTrainer.LbfgsPoissonRegression, RegressionTrainer.OnlineGradientDescent }
    .ForEach(x => experimentSettings.Trainers.Remove(x));
var experiment = ctx.Auto().CreateRegressionExperiment(experimentSettings);
var progressHandler = new RegressionExperimentProgressHandler();

ConsoleHelper.ConsoleWriteHeader("=============== Running AutoML experiment ===============");
Console.WriteLine($"Running AutoML regression experiment...");
var stopwatch = Stopwatch.StartNew();
var experimentResult = experiment.Execute(trainData: trainDataView, columnInformation: columnInfo, progressHandler: progressHandler);
Console.WriteLine($"{experimentResult.RunDetails.Count()} models were returned after {stopwatch.Elapsed.TotalSeconds:0.00} seconds{Environment.NewLine}");
Console.WriteLine("Top models ranked by root mean squared error --");
ConsoleHelper.PrintRegressionMetricsHeader();
var i = 0;
foreach (var run in experimentResult.RunDetails
    .Where(r => r.ValidationMetrics != null && !double.IsNaN(r.ValidationMetrics.RootMeanSquaredError))
    .OrderBy(r => r.ValidationMetrics.RootMeanSquaredError).Take(10))
{
    ConsoleHelper.PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
    i++;
}

ConsoleHelper.ConsoleWriteHeader("===== Evaluating model's accuracy with test data =====");
var predictions = experimentResult.BestRun.Model.Transform(testDataView);
var metrics = ctx.Regression.Evaluate(predictions, labelColumnName: "SalePrice", scoreColumnName: "Score");
ConsoleHelper.PrintRegressionMetrics(experimentResult.BestRun.TrainerName, metrics);

ConsoleHelper.ConsoleWriteHeader("=============== Saving the model ===============");
ctx.Model.Save(experimentResult.BestRun.Model, trainDataView.Schema, "./model.zip");
Console.WriteLine($"The model is saved as model.zip");

File.Delete(trainDataPath);
File.Delete(testDataPath);