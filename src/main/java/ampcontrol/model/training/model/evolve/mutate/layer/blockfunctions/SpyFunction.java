package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Creates a {@link SpyBlock} using {@link LayerBlockConfig}s from a source function.
 */
public class SpyFunction implements Function<Long, LayerBlockConfig> {

    private final UnaryOperator<GraphBuilderAdapter> spyFactory;
    private final Function<Long, LayerBlockConfig> source;

    public SpyFunction(UnaryOperator<GraphBuilderAdapter> spyFactory, Function<Long, LayerBlockConfig> source) {
        this.spyFactory = spyFactory;
        this.source = source;
    }

    @Override
    public LayerBlockConfig apply(Long nOut) {
        return new SpyBlock(source.apply(nOut)).setFactory(spyFactory);
    }

    /**
     * Create a new SpyFunction which sets weight init of all created {@link FeedForwardLayer}
     * @param source source function
     * @param weightInit {@link WeightInit} to set
     * @return a new SpyFunction
     */
    public static SpyFunction weightInit(Function<Long, LayerBlockConfig> source, WeightInit weightInit) {
        return new SpyFunction(factory -> new LayerSpyAdapter((layerName, layer, layerInputs) -> {
            if (layer instanceof BaseLayer) {
                ((BaseLayer) layer).setWeightInitFn(weightInit.getWeightInitFunction());
            }
        }, factory), source);
    }
}
