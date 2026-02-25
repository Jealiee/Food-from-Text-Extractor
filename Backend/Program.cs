using System.Diagnostics;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () =>
{
    var psi = new ProcessStartInfo
    {
        FileName = "python",
        Arguments = "-u -m model.main",
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
        WorkingDirectory = @"C:\Users\julka\Repos\Food from Text Extractor"
    };
    var process = Process.Start(psi);

    if (process == null)
    {
        return Results.Text("Failed to start Python process");
    }

    string errors = process.StandardError.ReadToEnd();
    string output = process.StandardOutput.ReadToEnd();
    process.WaitForExit();

    if (process.ExitCode != 0)
    {
        return Results.Text($"Python Error:\n{errors}");
    }

    return Results.Text(output);
});

app.Run();
