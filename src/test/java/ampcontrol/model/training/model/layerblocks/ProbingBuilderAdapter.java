package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test {@link BuilderAdapter} which is configured to expect input.
 *
 * @author Christian Skärby
 */
class ProbingBuilderAdapter implements BuilderAdapter {

    private int nextExpectedLayerInd;
    private final Function<Layer, Boolean> layerChecker;

    public ProbingBuilderAdapter(int nextExpectedLayerInd, Function<Layer, Boolean> layerChecker) {
        this.nextExpectedLayerInd = nextExpectedLayerInd;
        this.layerChecker = layerChecker;
    }

    @Override
    public LayerBlockConfig.BlockInfo layer(LayerBlockConfig.BlockInfo info, Layer layer) {
        assertEquals("Incorrect number of inputs!", 1 , info.getInputsNames().length);
        assertEquals("Incorrect layerInd!", nextExpectedLayerInd, info.getPrevLayerInd());
        assertTrue("Incorrect vertex type!", layerChecker.apply(layer));
        nextExpectedLayerInd++;
        return new LayerBlockConfig.SimpleBlockInfo.Builder(info).setPrevLayerInd(nextExpectedLayerInd).build();
    }

    private void assertNrofLayers(int expected) {
        assertEquals("Incorrect layerInd!", expected, nextExpectedLayerInd);
    }

    /**
     * Test that a given {@link LayerBlockConfig} interacts with a {@link BuilderAdapter} in a certain way.
     * @param layerChecker Tells if a given layer is expected or not
     * @param toTest {@link LayerBlockConfig} to test
     * @param expectedNrofLayers Expected number of layers
     */
    public static void testLayerBlock(Function<Layer, Boolean> layerChecker, LayerBlockConfig toTest, int expectedNrofLayers) {
        final int prevLayerInd = 666;

        final LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setPrevLayerInd(prevLayerInd)
                .setInputs(new String[] {"hyjkoa"})
                .build();
        final ProbingBuilderAdapter adapter = new ProbingBuilderAdapter(prevLayerInd, layerChecker);
        final LayerBlockConfig.BlockInfo output = toTest.addLayers(adapter,info);

        adapter.assertNrofLayers(prevLayerInd+expectedNrofLayers);
        assertEquals("Incorrect layerInd!", prevLayerInd+expectedNrofLayers, output.getPrevLayerInd());
    }

    /**
     * A queue of {@link Function Functions}.
     * @param <T>
     * @param <R>
     */
    static class FunctionQueue<T,R> implements Function<T,R> {

        private final Queue<Function<T,R>> queue;

        @SafeVarargs
        FunctionQueue(Function<T,R>... funcs) {
            queue = Stream.of(funcs).collect(Collectors.toCollection(ConcurrentLinkedQueue::new));
        }

        @Override
        public R apply(T t) {
            return queue.poll().apply(t);
        }
    }
}
