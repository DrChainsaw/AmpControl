package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;

/**
 * "Empty" {@link LayerBlockConfig} for testing
 *
 * @author Christian Skärby
 */
public class MockBlock implements LayerBlockConfig {

    private final int nrToIncrement;

    public MockBlock() {
        this(0);
    }

    public MockBlock(int nrToIncrement) {
        this.nrToIncrement = nrToIncrement;
    }

    @Override
    public String name() {
        return "";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        final int nextLayerInd = info.getPrevLayerInd()+nrToIncrement;
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(nextLayerInd)
                .setInputs(new String[] {info.getName(String.valueOf(nextLayerInd))})
                .build();
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        final int nextLayerInd = info.getPrevLayerInd()+nrToIncrement;
        graphBuilder.addLayer(info.getName(String.valueOf(nextLayerInd)), null, info.getInputsNames());
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(nextLayerInd)
                .setInputs(new String[] {info.getName(String.valueOf(nextLayerInd))})
                .build();
    }
}
