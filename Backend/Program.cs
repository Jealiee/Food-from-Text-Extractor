using System.Diagnostics;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () =>
{
    var psi = new ProcessStartInfo
    {
        FileName = "python",
        Arguments = "-u ../model/main.py",
        RedirectStandardOutput = true,
        UseShellExecute = false,
        CreateNoWindow = true
    };
    var process = Process.Start(psi);
    string output = process.StandardOutput.ReadToEnd();
    process.WaitForExit();

    return output;
});

app.Run();
