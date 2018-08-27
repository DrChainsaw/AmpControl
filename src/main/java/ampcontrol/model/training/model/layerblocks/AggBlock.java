package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Composition of two {@link LayerBlockConfig LayerBlockConfigs} which will operate on given {@link BuilderAdapter
 * BuilderAdapters} in sequence. Compositions of several {@link AggBlock AggBlocks} can be used to create complex
 * stacked architectures.
 *
 * @author Christian Skäeby
 */
public class AggBlock implements LayerBlockConfig {

    private final LayerBlockConfig first;
    private final String sep;
    private LayerBlockConfig then;

    /**
     * Constructor
     * @param first First {@link LayerBlockConfig}
     */
    public AggBlock(LayerBlockConfig first) {
        this(first, "_t_");
    }

    /**
     * Constructor
     * @param first First {@link LayerBlockConfig}
     * @param sep Separator string between names of {@link LayerBlockConfig LayerBlockConfigs} in composition.
     */
    public AggBlock(LayerBlockConfig first, String sep) {
        this.first = first;
        this.sep = sep;
    }

    @Override
    public String name() {
        String first = this.first.name();
        String then = this.then.name();
        String sepToUse = sep;

        if(first.isEmpty() || then.isEmpty()) {
            sepToUse = "";
        }

        return first + sepToUse + then;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        BlockInfo infoInner = first.addLayers(builder, info);
        return then.addLayers(builder, infoInner);
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        BlockInfo infoInner = first.addLayers(graphBuilder, info);
        return then.addLayers(graphBuilder, infoInner);
    }

    @Override
    public BlockInfo addLayers(NeuralNetConfiguration.ListBuilder listBuilder, BlockInfo info) {
        BlockInfo infoInner = first.addLayers(listBuilder, info);
        return then.addLayers(listBuilder, infoInner);
    }

    @Override
    public BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, BlockInfo info) {
        BlockInfo infoInner = first.addLayers(graphBuilder, info);
        return then.addLayers(graphBuilder, infoInner);
    }

    /**
     * Adds the second {@link LayerBlockConfig}
     * @param then the second {@link LayerBlockConfig}
     * @return the {@link AggBlock}
     */
    public AggBlock andThen(LayerBlockConfig then) {
        if(this.then != null) {
            throw new IllegalStateException("Block overwrite attempted!");
        }
        this.then = then;
        return this;
    }
}
