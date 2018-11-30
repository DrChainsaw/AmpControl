package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.graph.ResBlock;

import java.util.function.Function;

/**
 * Creates a {@link ResBlock} using {@link LayerBlockConfig}s from a source function.
 *
 * @author Christian Skärby
 */
public class ResBlockFunction implements Function<Long, LayerBlockConfig> {

    private final Function<Long, LayerBlockConfig> source;

    public ResBlockFunction(Function<Long, LayerBlockConfig> source) {
        this.source = source;
    }

    @Override
    public LayerBlockConfig apply(Long nOut) {
        return new ResBlock().setBlockConfig(source.apply(nOut));
    }
}
