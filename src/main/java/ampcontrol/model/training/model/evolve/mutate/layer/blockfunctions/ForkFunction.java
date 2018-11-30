package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.graph.ForkAgg;

import java.util.function.Function;
import java.util.function.IntSupplier;

/**
 * Creates a {@link ForkAgg} using {@link LayerBlockConfig}s from a source function. Number of paths in fork is supplied
 * through an {@link IntSupplier}. Actual number of paths might be smaller than provided value in case provided nrof paths
 * > nOut.
 *
 * @author Christian Skärby
 */
public class ForkFunction implements Function<Long, LayerBlockConfig> {

    private final IntSupplier nrofPathsSupplier;
    private final Function<Long, LayerBlockConfig> source;

    public ForkFunction(IntSupplier nrofPathsSupplier, Function<Long, LayerBlockConfig> source) {
        this.nrofPathsSupplier = nrofPathsSupplier;
        this.source = source;
    }

    @Override
    public LayerBlockConfig apply(Long nOut) {
        final long nrofPaths = Math.min(nOut, nrofPathsSupplier.getAsInt());
        long reminder = nOut;
        final ForkAgg fork = new ForkAgg();
        for (int pathInd = 0; pathInd < nrofPaths; pathInd++) {
            final long thisNout = reminder / (nrofPaths - pathInd);
            reminder -= thisNout;
            fork.add(source.apply(thisNout));
        }
        return fork;
    }
}
