using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using OxyPlot;
using OxyPlot.Series;

var mlContext = new MLContext();

// Simulação de 90 dias de saldo (últimos 3 meses)
var random = new Random();
var saldos = new List<SaldoEstoque>();
for (int i = 0; i < 90; i++)
{
    saldos.Add(new SaldoEstoque { Saldo = 1000 + random.Next(-100, 100) });
}

var dataView = mlContext.Data.LoadFromEnumerable(saldos);

// Configura o modelo SSA (Singular Spectrum Analysis)
var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedSaldo",
    inputColumnName: "Saldo",
    windowSize: 30, // tamanho da janela usada para decompor a série
    seriesLength: 90, // tamanho da série histórica
    trainSize: 90,
    horizon: 30); // número de dias a prever

var model = forecastingPipeline.Fit(dataView);

// Cria o forecast engine
var forecastEngine = model.CreateTimeSeriesEngine<SaldoEstoque, PrevisaoSaldo>(mlContext);

// Prever os próximos 30 dias
var forecast = forecastEngine.Predict();

Console.WriteLine("Previsão para os próximos 30 dias:\n");

for (int i = 0; i < saldos.Count; i++)
{
    var dataPrevista = DateTime.Today.AddDays(i - 89);
    Console.WriteLine($"{dataPrevista:yyyy-MM-dd}: R$ {saldos[i].Saldo:F2}");
}

for (int i = 0; i < forecast.ForecastedSaldo.Length; i++)
{
    var dataPrevista = DateTime.Today.AddDays(i + 1);
    Console.WriteLine($"{dataPrevista:yyyy-MM-dd}: R$ {forecast.ForecastedSaldo[i]:F2}");
}

// Criar o modelo de gráfico
var plotModel = new PlotModel { Title = "Gráfico de Exemplo" };

// Criar uma série de dados (como uma linha)
var lineSeries = new LineSeries
{
    Title = "Dados Base",
    MarkerType = MarkerType.Circle
};

var lineSeries2 = new LineSeries
{
    Title = "Dados Preditos",
    MarkerType = MarkerType.Diamond
};

// Adicionar pontos à série
lineSeries.Points.AddRange(saldos.Select((x, y) => new DataPoint(y + 1, x.Saldo)));
lineSeries2.Points.AddRange(forecast.ForecastedSaldo.Select((x, y) => new DataPoint(y + 91, x)));


// Adicionar a série ao modelo de gráfico
plotModel.Series.Add(lineSeries);
plotModel.Series.Add(lineSeries2);

// Exportar o gráfico para um arquivo PNG
using var stream = File.Create("grafico.png");
var exporter = new OxyPlot.SkiaSharp.PngExporter { Width = 800, Height = 600 };
exporter.Export(plotModel, stream);

Console.WriteLine("Gráfico exportado como 'grafico.png'");

public class SaldoEstoque
{
    public float Saldo { get; set; }
}

public class PrevisaoSaldo
{
    [VectorType(30)]
    public float[] ForecastedSaldo { get; set; }
}