using System.Reflection;

namespace StarlingHq.Tokenizers.Tests;

public class IndependenceTests
{
    [Fact]
    public void ProductionAssemblyDoesNotReferenceMicrosoftMLTokenizers()
    {
        var assembly = Assembly.Load("StarlingHq.Tokenizers");

        var references = assembly.GetReferencedAssemblies().Select(a => a.Name);

        Assert.DoesNotContain("Microsoft.ML.Tokenizers", references);
    }
}
