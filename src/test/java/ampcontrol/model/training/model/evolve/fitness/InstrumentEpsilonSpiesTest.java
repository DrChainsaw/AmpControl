package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.model.EvolvingGraphAdapter;
import ampcontrol.model.training.model.evolve.mutate.state.NoMutationStateWapper;
import ampcontrol.model.training.model.evolve.selection.ModelComparatorRegistry;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

/**
 * Test case for {@link InstrumentEpsilonSpies}
 *
 * @author Christian Skärby
 */
public class InstrumentEpsilonSpiesTest {

    /**
     * Test that a graph is instrumented and that fit method works
     */
    @Test
    public void instrumentAndFit() {
        final INDArray arr = Nd4j.randn(new long[]{1, 2, 10, 15});
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .setInputTypes(InputType.inferInputType(arr))
                .addLayer("first", new Convolution2D.Builder().nOut(7).build(), "input")
                .addVertex("firstSpy", new EpsilonSpyVertex(), "first")
                .addLayer("gp", new GlobalPoolingLayer(), "firstSpy")
                .addLayer("second", new DenseLayer.Builder().nOut(11).build(), "gp")
                .addLayer("third", new DenseLayer.Builder().nOut(13).build(), "second")
                .addVertex("thirdSpy", new EpsilonSpyVertex(), "third")
                .addLayer("output", new OutputLayer.Builder().nOut(1)
                        .lossFunction(new LossMSE())
                        .activation(new ActivationIdentity()).build(), "thirdSpy")
                .build());

        final ModelComparatorRegistry registry = new ModelComparatorRegistry();
        final InstrumentEpsilonSpies<EvolvingGraphAdapter> instrument = new InstrumentEpsilonSpies<>(registry);

        instrument.apply(EvolvingGraphAdapter.builder(graph).mutation(new NoMutationStateWapper<>(gb -> gb)).build(),
                fitness -> fail("Shall not report fitness!"));

        assertTrue("Expected comparator!", registry.get(graph).apply("first").isPresent());
        assertFalse("Did not expect comparator!", registry.get(graph).apply("second").isPresent());
        assertTrue("Expected comparator!", registry.get(graph).apply("third").isPresent());

        graph.fit(new DataSet(arr, Nd4j.ones(1)));
    }
}