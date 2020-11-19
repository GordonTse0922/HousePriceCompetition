using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using HousePricePrediction;
using Microsoft.ML;
using Resourcer;

var ctx = new MLContext();
var path = await ModelBuilder.GetTestDataFilePath();
var data = ctx.Data.LoadFromTextFile<ModelInput>(path: path, separatorChar: ',', hasHeader: true);
await using var model = Resource.AsStream("MLModel.zip");
var loadedModel = ctx.Model.Load(model, out var modelInputSchema);
var output = loadedModel.Transform(data);
var predictionOutcomes = ctx.Data.CreateEnumerable<ModelOutput>(output, false);

var csv = $@"Id,SalePrice
{string.Join("\n", predictionOutcomes.Select(x => x.ToString()))}
";

await File.WriteAllTextAsync("./outcome.csv", csv);
File.Delete(path);
Console.WriteLine("Done");
