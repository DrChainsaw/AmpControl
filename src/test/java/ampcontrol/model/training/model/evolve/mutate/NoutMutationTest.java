package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.GraphUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link NoutMutation}
 *
 * @author Christian Skärby
 */
public class NoutMutationTest {

    /**
     * Test mutation function
     */
    @Test
    public void mutate() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final NoutMutation noutMutation = new NoutMutation(() -> Stream.of(
                NoutMutation.NoutMutationDescription.builder().layerName(mut1).mutateNout(nOut -> 2 * nOut).build(),
                NoutMutation.NoutMutationDescription.builder().layerName(mut2).mutateNout(nOut -> 2 * nOut).build()));
        final ComputationGraph newGraph = new ComputationGraph(noutMutation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut1), newGraph.layerSize(mut1));
        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut2), newGraph.layerSize(mut2));
        assertEquals("Incorrect nOut!", graph.layerSize(noMut), newGraph.layerSize(noMut));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut in a residual conv layer followed by batchnorm and another residual conv layer.
     */
    @Test
    public void mutateResNet() {
        final String mut1 = "firstResConvToMutate";
        final String mut2 = "secondResConvToMutateToo";
        final String noMut = "afterMutate";
        final ComputationGraph graph = GraphUtils.getResNet(mut1, mut2, noMut);

        final long newNoutFirst = 5;
        final long newNoutSecond = 9;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
//                        NoutMutation.NoutMutationDescription.builder()
//                                .layerName(mut1)
//                                .mutateNout(nOut -> newNoutFirst)
//                                .build(),
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(mut2)
                                .mutateNout(nOut -> newNoutSecond)
                                .build()
                ))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(mut1));
        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(mut2));
        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(noMut));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test to decrease nOut in a convolution layer which is part of a fork.
     */
    @Test
    public void mutateForkNet() {
        final String beforeFork = "beforeFork";
        final String afterFork = "afterFork";
        final String fork1 = "fork1";
        final String fork2ToMutate = "fork2ToMutate";
        final String fork3 = "fork3";
        final ComputationGraph graph = GraphUtils.getForkNet(beforeFork, afterFork, fork1, fork2ToMutate, fork3);

        final long newNoutSecond = 9;
        final ComputationGraph newGraph = new ComputationGraph(new NoutMutation(
                () -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(fork2ToMutate)
                                .mutateNout(nOut -> newNoutSecond)
                                .build()))
                .mutate(
                        new ComputationGraphConfiguration.GraphBuilder(
                                graph.getConfiguration(),
                                new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", newNoutSecond, newGraph.layerSize(fork2ToMutate));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }
}