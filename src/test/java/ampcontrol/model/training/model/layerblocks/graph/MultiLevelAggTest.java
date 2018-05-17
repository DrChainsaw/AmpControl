package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MultiLevelAgg}
 *
 * @author Christian Skärby
 */
public class MultiLevelAggTest {

    /**
     * Test that name is consistent with configuration
     */
    @Test
    public void name() {
        final String sep = "grgt";
        final MultiLevelAgg mla = new MultiLevelAgg(sep);
        final String emptyName = mla.name();
        final List<LayerBlockConfig> sequence = IntStream.range(0,17)
                .mapToObj(i -> new NamedMockBlock("mb" + i))
                .peek(mb -> mla.andThen(mb))
                .collect(Collectors.toList());
        // Assumes begin and end part of emptyName are of equal length.
        final String nameStr = mla.name()
                .replace(emptyName.substring(0, emptyName.length()/2), "")
                .replace(emptyName.substring(emptyName.length()/2, emptyName.length()), "");
        final String[] names = nameStr.split(sep);
        assertEquals("Incorrect number of names!", sequence.size(), names.length);
        for(int i = 0; i < names.length; i++) {
            assertEquals("Incorrect name!", sequence.get(i).name(), names[i]);
        }


    }

    /**
     * Test that input to next layer is as expected.
     */
    @Test
    public void addLayers() {
        final int nrofLayerBlocksInMla = 11;
        final int nrofLayersPerMockBlock = 7;
        final MultiLevelAgg mla = new MultiLevelAgg();
        IntStream.range(0, nrofLayerBlocksInMla)
                .forEach(i -> mla.andThen(new MockBlock(nrofLayersPerMockBlock)));
        final LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setPrevLayerInd(666)
                .setInputs(new String[] {"input"})
                .setNameMapper(str -> "testMla_" + str)
                .build();

        final LayerBlockConfig.BlockInfo outInfo = mla.addLayers(new MockGraphAdapter(), info);
        for(int i = 0; i < nrofLayerBlocksInMla; i++) {
            // Assumption about internal of MultiLevelAgg: It uses GlobMeanMax
            final int expectedPrevLayer = info.getPrevLayerInd() + (i+1) * nrofLayersPerMockBlock;
            final LayerBlockConfig.BlockInfo expectedInfo = new GlobMeanMax()
                    .addLayers(
                            new MockGraphAdapter(),
                            new LayerBlockConfig.SimpleBlockInfo.Builder(info)
                    .setPrevLayerInd(expectedPrevLayer)
                    .build());
            assertEquals("Incorrect input name!", expectedInfo.getInputsNames()[0], outInfo.getInputsNames()[i]);

        }
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

}