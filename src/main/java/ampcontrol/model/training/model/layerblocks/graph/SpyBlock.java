package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Allows for spying on information about added layers through a {@link GraphSpyAdapter.LayerSpy}.
 *
 * @author Christian Skärby
 */
public class SpyBlock implements LayerBlockConfig {

    private final LayerBlockConfig blockConfig;
    private final GraphSpyAdapter.LayerSpy spy;

    public SpyBlock(LayerBlockConfig blockConfig, GraphSpyAdapter.LayerSpy spy) {
        this.blockConfig = blockConfig;
        this.spy = spy;
    }

    @Override
    public String name() {
        return blockConfig.name();
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Only implemented for graphs!");
    }

    @Override
    public BlockInfo addLayers(NeuralNetConfiguration.ListBuilder listBuilder, BlockInfo info) {
        throw new IllegalArgumentException("Only implemented for graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        final GraphSpyAdapter spyAdapter = new GraphSpyAdapter(graphBuilder, spy);
        return blockConfig.addLayers(spyAdapter, info);
    }
}
