package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.Dense;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.MockGraphAdapter;
import org.apache.commons.lang.mutable.MutableInt;
import org.junit.Test;

import java.util.function.Function;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ForkFunction}
 *
 * @author Christian Skärby
 */
public class ForkFunctionTest {

    /**
     * Test zero edge case
     */
    @Test
    public void applyZero() {
        final Function<Long, LayerBlockConfig> forkFun = new ForkFunction(() -> 0, nOut -> new Dense());
        MutableInt cnt = new MutableInt(0);
        forkFun.apply(4L).addLayers(new LayerSpyAdapter(
                        (layerName, layer, layerInputs) -> cnt.increment(), new MockGraphAdapter()),
                new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"dummy"}).build());
        assertEquals("Incorrect number of paths!", 0, cnt.getValue());
    }

    /**
     * Test that the supplied number of paths are created
     */
    @Test
    public void apply() {
        final Function<Long, LayerBlockConfig> forkFun = new ForkFunction(() -> 3, nOut -> new Dense());
        MutableInt cnt = new MutableInt(0);
        forkFun.apply(4L).addLayers(new LayerSpyAdapter(
                (layerName, layer, layerInputs) -> cnt.increment(), new MockGraphAdapter()),
                new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"dummy"}).build());
        assertEquals("Incorrect number of paths!", 3, cnt.getValue());
    }

    /**
     * Test that no more than nOut number of paths are created
     */
    @Test
    public void applyLimited() {
        final Function<Long, LayerBlockConfig> forkFun = new ForkFunction(() -> 4, nOut -> new Dense());
        MutableInt cnt = new MutableInt(0);
        forkFun.apply(3L).addLayers(new LayerSpyAdapter(
                        (layerName, layer, layerInputs) -> cnt.increment(), new MockGraphAdapter()),
                new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"dummy"}).build());
        assertEquals("Incorrect number of paths!", 3, cnt.getValue());
    }
}