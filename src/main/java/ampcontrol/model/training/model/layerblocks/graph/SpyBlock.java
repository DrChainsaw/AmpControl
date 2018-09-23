package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import java.util.function.UnaryOperator;

/**
 * Allows for spying on information about added layers through a {@link GraphSpyAdapter.LayerSpy}.
 *
 * @author Christian Skärby
 */
public class SpyBlock implements LayerBlockConfig {

    private final LayerBlockConfig blockConfig;
    private UnaryOperator<GraphBuilderAdapter> factory = UnaryOperator.identity();

    public SpyBlock(LayerBlockConfig blockConfig) {
        this.blockConfig = blockConfig;
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
        final GraphBuilderAdapter spyAdapter = factory.apply(graphBuilder);
        return blockConfig.addLayers(spyAdapter, info);
    }

    public SpyBlock setFactory(UnaryOperator<GraphBuilderAdapter> factory) {
        this.factory = factory;
        return this;
    }
}
