using System.Reflection;

namespace Starling.Tokenizers.Tests;

public class IndependenceTests
{
    [Fact]
    public void ProductionAssemblyDoesNotReferenceMicrosoftMLTokenizers()
    {
        var assembly = Assembly.Load("Starling.Tokenizers");

        var references = assembly.GetReferencedAssemblies().Select(a => a.Name);

        Assert.DoesNotContain("Microsoft.ML.Tokenizers", references);
    }
}
