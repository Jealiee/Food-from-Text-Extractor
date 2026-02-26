using System.Net.Http;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/predict", async (string text) =>
{
    using var client = new HttpClient();
    var url = $"http://127.0.0.1:8000/predict?text={Uri.EscapeDataString(text)}";

    try
    {
        var response = await client.GetStringAsync(url);
        return Results.Text(response);
    }

    catch (Exception ex)
    {
        return Results.Text($"Error calling Python API: {ex.Message}");
    }
});

app.Run();
