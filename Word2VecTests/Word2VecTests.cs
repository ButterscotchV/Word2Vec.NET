using System;
using System.Collections.Generic;
using System.Text;
using Word2Vec.NeuralNetwork;
using Xunit;
using Xunit.Abstractions;

namespace Word2Vec.Tests
{
    public class Word2VecTests
    {
        private readonly ITestOutputHelper output;

        public Word2VecTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        public class XUnitTrainingProgressListener : TrainingProgressListener
        {
            private readonly ITestOutputHelper output;

            public XUnitTrainingProgressListener(ITestOutputHelper output)
            {
                this.output = output;
            }

            public override void Update(Stage stage, double progress)
            {
                output.WriteLine($"Stage {Enum.GetName(stage)}, progress {progress*100.0:0.00}%");
            }
        }

        List<string> Sentences = new List<string>()
        {
            "hello this is a test",
            "testing this is a very difficult process",
            "and i would like for it to work",
            "so please test this so it works"
        };

        IEnumerable<List<string>> SplitStrings(List<string> sentences)
        {
            foreach (var sentence in sentences)
            {
                yield return new List<string>(sentence.Split(' '));
            }
        }

        [Fact()]
        public void TrainTest()
        {
            var builder = Word2VecModel<string>.Trainer();

            builder.MinVocabFrequency = 1;
            builder.WindowSize = 32;
            builder.Type = NeuralNetworkTypeValues.SKIP_GRAM;
            builder.UseHierarchicalSoftmax = true;
            builder.LayerSize = 128;
            builder.Iterations = 3;
            builder.Listener = new XUnitTrainingProgressListener(output);

            var model = builder.Train(SplitStrings(Sentences));

            Assert.Equal(builder.LayerSize, model.LayerSize);

            string[] vocab = model.Vocab;
            double[][] vectors = model.Vectors;

            Assert.True(vocab.Length > 0, "Vocab is empty");
            Assert.Equal(vocab.Length, vectors.Length);
            Assert.Equal(model.LayerSize, vectors[0].Length);

            var sb = new StringBuilder();
            for (var i = 0; i < vocab.Length; i++)
            {
                sb.AppendLine($"\"{vocab[i]}\": ({vectors[i][0]}, {vectors[i][1]}, ...)");
            }
            output.WriteLine(sb.ToString());
        }
    }
}
