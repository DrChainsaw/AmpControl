package ampcontrol.model.training.model.evolve.mutate;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.transferlearning.TransferLearning.GraphBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.UnaryOperator;
import java.util.stream.Stream;

/**
 * Mutation operation which changes nOut of layers
 *
 * @author Christian Skärby
 */
public class MutateNout implements Mutation {

    private static final Logger log = LoggerFactory.getLogger(MutateNout.class);

    private final Supplier<NoutMutation> mutationLayerSupplier;

    /**
     * Interface for mutating Nout
     */
    @Builder @Getter
    public static class NoutMutation {
        private final String layerName;
        private final UnaryOperator<Long> mutateNout;
    }


    public MutateNout(Supplier<NoutMutation> mutationLayerSupplier) {
        this.mutationLayerSupplier = mutationLayerSupplier;
    }

    @Override
    public GraphBuilder mutate(GraphBuilder builder, ComputationGraph prevGraph) {
        final Map<String, FeedForwardLayer> changedLayers = new HashMap<>(); // To be filled in
        mutationLayerSupplier.stream().forEach(mutation -> updateNoutOfLayer(changedLayers, builder, prevGraph, mutation));
        return builder;
    }

    private GraphBuilder updateNoutOfLayer(
            Map<String, FeedForwardLayer> changedLayers,
            GraphBuilder builder,
            ComputationGraph prevGraph,
            NoutMutation mutation) {
        final String layerName = mutation.getLayerName();
        final FeedForwardLayer newLayerConf = changedLayers.computeIfAbsent(layerName,
                key -> (FeedForwardLayer) prevGraph.getLayer(layerName).conf().getLayer().clone());
        newLayerConf.setNOut(mutation.getMutateNout().apply(newLayerConf.getNOut()));
        log.info("Mutating nOut of layer " + layerName + " from " + prevGraph.layerSize(layerName) + " to " + newLayerConf.getNOut());
        final List<String> inputs = prevGraph.getConfiguration().getVertexInputs().get(layerName);
        builder.removeVertexKeepConnections(layerName)
                .addLayer(layerName, newLayerConf, inputs.toArray(new String[0]));

        updateNinOfOutputLayer(changedLayers, builder, prevGraph, layerName, prevGraph.layerSize(layerName) - newLayerConf.getNOut());
        return builder;
    }

    private void updateNinOfOutputLayer(
            Map<String, FeedForwardLayer> changedLayers,
            GraphBuilder builder,
            ComputationGraph prevGraph,
            String layerName,
            long nNinDelta) {
        Stream.of(Optional.ofNullable(prevGraph.getVertex(layerName).getOutputVertices()).orElse(new VertexIndices[0]))
                .map(vertexInd -> prevGraph.getVertices()[vertexInd.getVertexIndex()])
                .filter(vertex -> !vertex.getVertexName().equals(layerName))
                .forEachOrdered(graphVertex -> {
                    if (graphVertex.hasLayer() && graphVertex.getLayer().conf().getLayer() instanceof FeedForwardLayer) { // Layer for which it is possible to set inputs

                        final FeedForwardLayer newLayerConf = changedLayers.computeIfAbsent(graphVertex.getVertexName(),
                                key -> (FeedForwardLayer) graphVertex.getLayer().conf().getLayer().clone());

                        final long newNIn = newLayerConf.getNIn() - nNinDelta;
                        newLayerConf.setNIn(newNIn);
                        if (Mutation.changeNinMeansChangeNout(graphVertex)) {
                            newLayerConf.setNOut(newNIn);
                        }

                        final List<String> vertexInputs = prevGraph.getConfiguration().getVertexInputs().get(newLayerConf.getLayerName());
                        builder.removeVertexKeepConnections(newLayerConf.getLayerName());

                        builder.addLayer(
                                newLayerConf.getLayerName(),
                                newLayerConf,
                                vertexInputs.toArray(new String[0]));
                    }
                    if (Mutation.doesNinPropagateToNext(graphVertex)) {
                        updateNinOfOutputLayer(changedLayers, builder, prevGraph, graphVertex.getVertexName(), nNinDelta);
                    }
                });
    }
}
