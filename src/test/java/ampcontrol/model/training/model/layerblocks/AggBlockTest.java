package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.graph.MockBlock;
import ampcontrol.model.training.model.layerblocks.graph.MockGraphAdapter;
import org.junit.Test;

import java.util.function.Supplier;

import static org.junit.Assert.*;

/**
 * Test cases for {@link AggBlock}
 *
 * @author Christian Skärby
 */
public class AggBlockTest {

    /**
     * Test name with two full {@link LayerBlockConfig LayerBlockConfigs}
     */
    @Test
    public void nameFull() {
        final String sep = "_then_";
        final LayerBlockConfig lbc1 = new NamedMockBlock("name1");
        final LayerBlockConfig lbc2 = new NamedMockBlock("name2");
        final String result = new AggBlock(lbc1, sep).andThen(lbc2).name();
        final String expected = lbc1.name() + sep + lbc2.name();
        assertEquals("Incorrect first part!", expected, result);
    }

    /**
     * Test name with only first {@link LayerBlockConfig} having a name
     */
    @Test
    public void nameFirstOnly() {
        final String sep = "_then_";
        final LayerBlockConfig lbc1 = new NamedMockBlock("name1");
        final LayerBlockConfig lbc2 = new NamedMockBlock("");
        final String result = new AggBlock(lbc1, sep).andThen(lbc2).name();
        final String expected = lbc1.name();
        assertEquals("Incorrect first part!", expected, result);
    }

    /**
     * Test name with only second {@link LayerBlockConfig} having a name
     */
    @Test
    public void nameSecondOnly() {
        final String sep = "_then_";
        final LayerBlockConfig lbc1 = new NamedMockBlock("");
        final LayerBlockConfig lbc2 = new NamedMockBlock("name2");
        final String result = new AggBlock(lbc1, sep).andThen(lbc2).name();
        final String expected = lbc2.name();
        assertEquals("Incorrect first part!", expected, result);
    }


    /**
     * Test add layers with a {@link BuilderAdapter}
     */
    @Test
    public void addLayersBuilderAdapter() {
        final Supplier<Boolean> seqChecker = new BoolTicToc();
        final SequenceCheckingBlock lbc1 = new SequenceCheckingBlock(seqChecker);
        final SequenceCheckingBlock lbc2 = new SequenceCheckingBlock(()-> !seqChecker.get());
        final BuilderAdapter adapter = new MockGraphAdapter();
        final LayerBlockConfig.BlockInfo input = new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(666).build();
        LayerBlockConfig.BlockInfo output = new AggBlock(lbc1).andThen(lbc2).addLayers(adapter, input);
        lbc1.assertWasCalled(true);
        lbc2.assertWasCalled(true);
        assertEquals("Incorrect output!", input.getPrevLayerInd()+2, output.getPrevLayerInd());
    }

    /**
     * Test addLayers with a {@link GraphBuilderAdapter}
     */
    @Test
    public void addLayersGraphAdapter() {
        final Supplier<Boolean> seqChecker = new BoolTicToc();
        final SequenceCheckingBlock lbc1 = new SequenceCheckingBlock(seqChecker);
        final SequenceCheckingBlock lbc2 = new SequenceCheckingBlock(()-> !seqChecker.get());
        final GraphBuilderAdapter adapter = new MockGraphAdapter();
        final LayerBlockConfig.BlockInfo input = new LayerBlockConfig.SimpleBlockInfo.Builder().setPrevLayerInd(666).build();
        LayerBlockConfig.BlockInfo output = new AggBlock(lbc1).andThen(lbc2).addLayers(adapter, input);
        lbc1.assertWasCalled(true);
        lbc2.assertWasCalled(true);
        assertEquals("Incorrect output!", input.getPrevLayerInd()+2, output.getPrevLayerInd());
    }

    private static class NamedMockBlock extends MockBlock {
        private final String name;

        public NamedMockBlock(String name) {
            this.name = name;
        }

        @Override
        public String name() {
            return name;
        }
    }

    private static class SequenceCheckingBlock implements LayerBlockConfig {

        private final Supplier<Boolean> expectCalled;
        private boolean wasCalled = false;

        private SequenceCheckingBlock(Supplier<Boolean> expectCalled) {
            this.expectCalled = expectCalled;
        }

        @Override
        public String name() {
            fail("Not expected!");
            return "";
        }

        @Override
        public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
            assertTrue("Incorrect sequence!", expectCalled.get());
            wasCalled = true;
            return new SimpleBlockInfo.Builder(info).setPrevLayerInd(info.getPrevLayerInd()+1).build();
        }

        @Override
        public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
            assertTrue("Incorrect sequence!", expectCalled.get());
            wasCalled = true;
            return new SimpleBlockInfo.Builder(info).setPrevLayerInd(info.getPrevLayerInd()+1).build();
        }

        private void assertWasCalled(boolean expected) {
            assertEquals("Incorrect state for was called!", expected, wasCalled);
        }
    }

    private static class BoolTicToc implements Supplier<Boolean> {
        private boolean state = false;

        @Override
        public Boolean get() {
            state = !state;
            return state;
        }
    }
}