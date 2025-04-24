using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Legends;
using OxyPlot.Series;
using System.Globalization;

var mlContext = new MLContext();

// Simulação de 90 dias de saldo (últimos 3 meses)
var random = new Random();
var saldos = new List<SaldoEstoque>();
var path = @"SEU PATH AQUI";
var data = File.ReadAllLines(@$"{path}\data.csv").Skip(1).Select(a => a.Split(","));
saldos.AddRange(data.Select(x => new SaldoEstoque { Data = DateTime.Parse(x[0].Replace("\"", "")), Saldo = float.Parse(x[2].Replace("\"", ""), CultureInfo.InvariantCulture) }));

var dataView = mlContext.Data.LoadFromEnumerable(saldos);


// Configura o modelo SSA (Singular Spectrum Analysis)
var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedSaldo",
    inputColumnName: "Saldo",
    windowSize: 30, // tamanho da janela usada para decompor a série
    seriesLength: saldos.Count, // tamanho da série histórica
    trainSize: saldos.Count,
    horizon: 30); // número de dias a prever

var model = forecastingPipeline.Fit(dataView);

// Cria o forecast engine
var forecastEngine = model.CreateTimeSeriesEngine<SaldoEstoque, PrevisaoSaldo>(mlContext);

// Prever os próximos 30 dias
var forecast = forecastEngine.Predict();

Console.WriteLine("Previsão para os próximos 30 dias:\n");

for (int i = 0; i < saldos.Count; i++)
{
    Console.WriteLine($"{saldos[i].Data:yyyy-MM-dd}: {saldos[i].Saldo:F2}");
}

var ultimaData = saldos.LastOrDefault().Data;

for (int i = 0; i < forecast.ForecastedSaldo.Length; i++)
{
    var dataPrevista = ultimaData.AddDays(i + 1);
    Console.WriteLine($"{dataPrevista:yyyy-MM-dd}: {forecast.ForecastedSaldo[i]:F2}");
}

// Criar o modelo de gráfico
var plotModel = new PlotModel
{
    Title = "Previsão de Estoque dos próximos 30 dias",
};

// Criar o modelo de gráfico
plotModel.Legends.Add(new Legend()
{
    LegendTitle = "Legenda"
});

plotModel.Axes.Add(new DateTimeAxis() { Position = AxisPosition.Bottom, Title = "Dias", StringFormat = "dd/MM/yyyy", IntervalType = DateTimeIntervalType.Days });
plotModel.Axes.Add(new LinearAxis() { Position = AxisPosition.Left, Title = "Qtde. Estoque" });

// Criar uma série de dados (como uma linha)
var lineSeries = new LineSeries
{
    Title = "Dados Base",
    MarkerType = MarkerType.Circle,
};

var lineSeries2 = new LineSeries
{
    Title = "Dados Preditos",
    MarkerType = MarkerType.Diamond
};

// Adicionar pontos à série
lineSeries.Points.AddRange(saldos.Select((x, y) => new DataPoint(DateTimeAxis.ToDouble(x.Data), x.Saldo)));
lineSeries2.Points.AddRange(forecast.ForecastedSaldo.Select((x, y) => new DataPoint(DateTimeAxis.ToDouble(ultimaData.AddDays(y + 1)), x)));

// Adicionar a série ao modelo de gráfico
plotModel.Series.Add(lineSeries);
plotModel.Series.Add(lineSeries2);

// Exportar o gráfico para um arquivo PNG
using var stream = File.Create("grafico.jpg");
var exporter = new OxyPlot.SkiaSharp.JpegExporter { Width = 800, Height = 600 };
exporter.Export(plotModel, stream);

Console.WriteLine("Gráfico exportado como 'grafico.jpg'");

public class SaldoEstoque
{
    public DateTime Data { get; set; }

    public float Saldo { get; set; }
}

public class PrevisaoSaldo
{
    [VectorType(30)]
    public float[] ForecastedSaldo { get; set; }
}