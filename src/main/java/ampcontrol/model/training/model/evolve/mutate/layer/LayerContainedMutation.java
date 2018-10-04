package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.function.Function;

/**
 * Inserts a layer in the graph. If a layer with the same name exists already it will be replaced. Note that for this
 * to work, the change of layer must be contained, i.e it must not "spill over" to other layers. Examples of what
 * "spills over" is changes in nIn or nOut as well as general changes in output shape in case layer is one of several
 * inputs to a vertex which require all their inputs to be of same shape.
 *
 * @author Christian Skärby
 */
public class LayerContainedMutation implements Mutation<GraphBuilder> {

    private static final Logger log = LoggerFactory.getLogger(LayerContainedMutation.class);

    private final Supplier<LayerMutation> mutationSupplier;

    @Getter
    @Builder
    public static class LayerMutation {
        private final LayerMutationInfo mutationInfo;
        private final Function<Layer, Layer> mutation;
    }

    public LayerContainedMutation(Supplier<LayerMutation> mutationSupplier) {
        this.mutationSupplier = mutationSupplier;
    }

    @Override
    public GraphBuilder mutate(GraphBuilder builder) {
        mutationSupplier.stream().forEach(mutation -> replaceLayer(mutation, builder));
        return builder;
    }

    private void replaceLayer(LayerMutation mutation, GraphBuilder builder) {

        final LayerMutationInfo mutationInfo = mutation.getMutationInfo();
        if (!Optional.ofNullable(builder.getVertices().get(mutationInfo.getLayerName())).isPresent()) {
            throw new IllegalArgumentException("Tried to mutate layer " + mutationInfo.getLayerName() + " but no such layer exists!");
        }

        final Layer toMutate = ((LayerVertex) builder.getVertices().get(mutationInfo.getLayerName())).getLayerConf().getLayer();

        final long nOut = mutationInfo.getOutputSizeMapping().apply(mutationInfo.getLayerName(), builder);
        final long nIn = mutationInfo.getInputSizeMapping().apply(mutationInfo.getLayerName(), builder);
        builder.removeVertex(mutationInfo.getLayerName(), false);

        final Layer mutatedLayer = mutation.getMutation().apply(toMutate);
        if (mutatedLayer instanceof FeedForwardLayer) {
            FeedForwardLayer ffLayer = ((FeedForwardLayer) mutatedLayer);
            ffLayer.setNIn(nIn);
            ffLayer.setNOut(nOut);
        }

        log.info("Mutated layer " + mutationInfo.getLayerName() + " to " + mutatedLayer);
        builder.addLayer(mutationInfo.getLayerName(), mutatedLayer, mutationInfo.getInputLayers());
    }
}
