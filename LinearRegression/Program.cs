using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Legends;
using OxyPlot.Series;
using System.Globalization;

var fnCreateFeatures = (List<SaldoEstoque> rawData) =>
{
    // Sort by date
    var sorted = rawData.OrderBy(x => x.Data).ToList();
    var result = new List<TimeSeriesSaldo>();

    for (int i = 7; i < sorted.Count; i++)
    {
        float lag1 = sorted[i - 1].Saldo;
        float lag7 = sorted[i - 7].Saldo;
        float rollingMean7 = sorted.Skip(i - 7).Take(7).Average(x => x.Saldo);

        var date = sorted[i].Data;

        result.Add(new TimeSeriesSaldo
        {
            Saldo = sorted[i].Saldo,
            Lag1 = lag1,
            Lag7 = lag7,
            RollingMean7 = rollingMean7,
            Year = date.Year,
            Month = date.Month,
            Day = date.Day,
            DayOfWeek = (float)date.DayOfWeek,
            IsWeekend = (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday) ? 1 : 0
        });
    }

    return result;
};

static List<(DateTime Date, float Forecast)> ForecastNextNDays(
    ITransformer model,
    MLContext context,
    List<SaldoEstoque> historicalData,
    int forecastDays)
{
    var predictionEngine = context.Model.CreatePredictionEngine<TimeSeriesSaldo, PredictionResult>(model);
    var results = new List<(DateTime, float)>();

    // Ensure historical data is sorted by date
    historicalData = historicalData.OrderBy(d => d.Data).ToList();

    for (int i = 0; i < forecastDays; i++)
    {
        var lastDate = historicalData.Last().Data;
        var nextDate = lastDate.AddDays(1);

        // Create lag and rolling features from historical data
        float lag1 = historicalData[historicalData.Count - 1].Saldo;
        float lag7 = historicalData.Count >= 7 ? historicalData[historicalData.Count - 7].Saldo : lag1;
        float rolling7 = historicalData.Count >= 7
            ? historicalData.Skip(historicalData.Count - 7).Take(7).Average(x => x.Saldo)
            : historicalData.Average(x => x.Saldo);

        var nextFeatures = new TimeSeriesSaldo
        {
            Lag1 = lag1,
            Lag7 = lag7,
            RollingMean7 = rolling7,
            Year = nextDate.Year,
            Month = nextDate.Month,
            Day = nextDate.Day,
            DayOfWeek = (float)nextDate.DayOfWeek,
            IsWeekend = (nextDate.DayOfWeek == DayOfWeek.Saturday || nextDate.DayOfWeek == DayOfWeek.Sunday) ? 1 : 0
        };

        // Predict
        var prediction = predictionEngine.Predict(nextFeatures);
        float predictedQuantity = prediction.Score;

        results.Add((nextDate, predictedQuantity));

        // Add the prediction as if it were real to update future lags
        historicalData.Add(new SaldoEstoque
        {
            Data = nextDate,
            Saldo = predictedQuantity
        });
    }

    return results;
}

var mlContext = new MLContext();

// Simulação de 90 dias de saldo (últimos 3 meses)
var random = new Random();
var saldos = new List<SaldoEstoque>();
var path = @"D:\Downloads";
var data = File.ReadAllLines(@$"{path}\data.csv")
    .Skip(1)
    .Select(a => a.Split(","));

saldos.AddRange(data.Select(x => new SaldoEstoque { Data = DateTime.Parse(x[0].Replace("\"", "")), Saldo = float.Parse(x[1].Replace("\"", ""), CultureInfo.InvariantCulture) }));

#region SSA
//var dataView = mlContext.Data.LoadFromEnumerable(saldos);

// Configura o modelo SSA (Singular Spectrum Analysis)
//var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
//    outputColumnName: "ForecastedSaldo",
//    inputColumnName: "Saldo",
//    windowSize: 30, // tamanho da janela usada para decompor a série
//    seriesLength: saldos.Count, // tamanho da série histórica
//    trainSize: saldos.Count,
//    horizon: 30); // número de dias a prever

//var model = forecastingPipeline.Fit(dataView);

//// Cria o forecast engine
//var forecastEngine = model.CreateTimeSeriesEngine<SaldoEstoque, PrevisaoSaldo>(mlContext);

//// Prever os próximos 30 dias
//var forecast = forecastEngine.Predict();

//Console.WriteLine("Previsão para os próximos 30 dias:\n");

//for (int i = 0; i < saldos.Count; i++)
//{
//    Console.WriteLine($"{saldos[i].Data:yyyy-MM-dd}: {saldos[i].Saldo:F2}");
//}

//var ultimaData = saldos.LastOrDefault().Data;

//for (int i = 0; i < forecast.ForecastedSaldo.Length; i++)
//{
//    var dataPrevista = ultimaData.AddDays(i + 1);
//    Console.WriteLine($"{dataPrevista:yyyy-MM-dd}: {forecast.ForecastedSaldo[i]:F2}");
//}

//// Criar o modelo de gráfico
//var plotModel = new PlotModel
//{
//    Title = "Previsão de Estoque dos próximos 30 dias",
//};

//// Criar o modelo de gráfico
//plotModel.Legends.Add(new Legend()
//{
//    LegendTitle = "Legenda"
//});

//plotModel.Axes.Add(new DateTimeAxis() { Position = AxisPosition.Bottom, Title = "Dias", StringFormat = "dd/MM/yyyy", IntervalType = DateTimeIntervalType.Days });
//plotModel.Axes.Add(new LinearAxis() { Position = AxisPosition.Left, Title = "Qtde. Estoque" });

//// Criar uma série de dados (como uma linha)
//var lineSeries = new LineSeries
//{
//    Title = "Dados Base",
//    MarkerType = MarkerType.Circle,
//};

//var lineSeries2 = new LineSeries
//{
//    Title = "Dados Preditos",
//    MarkerType = MarkerType.Diamond
//};

//// Adicionar pontos à série
//lineSeries.Points.AddRange(saldos.Select((x, y) => new DataPoint(DateTimeAxis.ToDouble(x.Data), x.Saldo)));
//lineSeries2.Points.AddRange(forecast.ForecastedSaldo.Select((x, y) => new DataPoint(DateTimeAxis.ToDouble(ultimaData.AddDays(y + 1)), x)));

//// Adicionar a série ao modelo de gráfico
//plotModel.Series.Add(lineSeries);
//plotModel.Series.Add(lineSeries2);

//// Exportar o gráfico para um arquivo PNG
//using var stream = File.Create("grafico.jpg");
//var exporter = new OxyPlot.SkiaSharp.JpegExporter { Width = 800, Height = 600 };
//exporter.Export(plotModel, stream);

//Console.WriteLine("Gráfico exportado como 'grafico.jpg'");
#endregion

var processedData = fnCreateFeatures(saldos);

var dataView = mlContext.Data.LoadFromEnumerable(processedData);

var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

var pipeline = mlContext.Transforms.Concatenate("Features",
        nameof(TimeSeriesSaldo.Lag1),
        nameof(TimeSeriesSaldo.Lag7),
        nameof(TimeSeriesSaldo.RollingMean7),
        nameof(TimeSeriesSaldo.Year),
        nameof(TimeSeriesSaldo.Month),
        nameof(TimeSeriesSaldo.Day),
        nameof(TimeSeriesSaldo.DayOfWeek),
        nameof(TimeSeriesSaldo.IsWeekend))
    .Append(mlContext.Regression.Trainers.LightGbm(labelColumnName: "Saldo"));

var model = pipeline.Fit(trainTestData.TrainSet);

var predictions = model.Transform(trainTestData.TestSet);

var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Saldo");

Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
Console.WriteLine($"MSE: {metrics.MeanSquaredError}");
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError}");

var forecast = ForecastNextNDays(model, mlContext, saldos, forecastDays: 5);

Console.WriteLine("Previsão para os próximos 5 dias:\n");

foreach (var day in forecast)
{
    Console.WriteLine($"{day.Date:yyyy-MM-dd}: {day.Forecast:N2}");
}

//for (int i = 0; i < saldos.Count; i++)
//{
//    Console.WriteLine($"{saldos[i].Data:yyyy-MM-dd}: {saldos[i].Saldo:F2}");
//}

//var ultimaData = saldos.LastOrDefault().Data;

//for (int i = 0; i < forecast.ForecastedSaldo.Length; i++)
//{
//    var dataPrevista = ultimaData.AddDays(i + 1);
//    Console.WriteLine($"{dataPrevista:yyyy-MM-dd}: {forecast.ForecastedSaldo[i]:F2}");
//}

//// Criar o modelo de gráfico
//var plotModel = new PlotModel
//{
//    Title = "Previsão de Estoque dos próximos 30 dias",
//};

//// Criar o modelo de gráfico
//plotModel.Legends.Add(new Legend()
//{
//    LegendTitle = "Legenda"
//});

//plotModel.Axes.Add(new DateTimeAxis() { Position = AxisPosition.Bottom, Title = "Dias", StringFormat = "dd/MM/yyyy", IntervalType = DateTimeIntervalType.Days });
//plotModel.Axes.Add(new LinearAxis() { Position = AxisPosition.Left, Title = "Qtde. Estoque" });

//// Criar uma série de dados (como uma linha)
//var lineSeries = new LineSeries
//{
//    Title = "Dados Base",
//    MarkerType = MarkerType.Circle,
//};

//var lineSeries2 = new LineSeries
//{
//    Title = "Dados Preditos",
//    MarkerType = MarkerType.Diamond
//};

//// Adicionar pontos à série
//lineSeries.Points.AddRange(saldos.Select((x, y) => new DataPoint(DateTimeAxis.ToDouble(x.Data), x.Saldo)));
//lineSeries2.Points.AddRange(forecast.ForecastedSaldo.Select((x, y) => new DataPoint(DateTimeAxis.ToDouble(ultimaData.AddDays(y + 1)), x)));

//// Adicionar a série ao modelo de gráfico
//plotModel.Series.Add(lineSeries);
//plotModel.Series.Add(lineSeries2);

//// Exportar o gráfico para um arquivo PNG
//using var stream = File.Create("grafico.jpg");
//var exporter = new OxyPlot.SkiaSharp.JpegExporter { Width = 800, Height = 600 };
//exporter.Export(plotModel, stream);

//Console.WriteLine("Gráfico exportado como 'grafico.jpg'");

public class SaldoEstoque
{
    public DateTime Data { get; set; }

    public float Saldo { get; set; }
}

public class TimeSeriesSaldo
{
    public float Saldo { get; set; }
    public float Lag1 { get; set; }
    public float Lag7 { get; set; }
    public float RollingMean7 { get; set; }
    public float Year { get; set; }
    public float Month { get; set; }
    public float Day { get; set; }
    public float DayOfWeek { get; set; }
    public float IsWeekend { get; set; }
}

public class PrevisaoSaldo
{
    [VectorType(30)]
    public float[] ForecastedSaldo { get; set; }
}

public class PredictionResult
{
    public float Score { get; set; }
}