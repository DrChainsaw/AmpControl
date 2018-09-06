package ampcontrol.model.training.model.mutate;

import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.transferlearning.TransferLearning.GraphBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.IntUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Mutation operation which changes nOut of layers
 *
 * @author Christian Skärby
 */
public class MutateNout implements Mutation {

    private final Supplier<Stream<String>> mutationLayerSupplier;
    private final IntUnaryOperator mutationFunction;

    public MutateNout(Supplier<Stream<String>> mutationLayerSupplier, IntUnaryOperator mutationFunction) {
        this.mutationLayerSupplier = mutationLayerSupplier;
        this.mutationFunction = mutationFunction;
    }

    @Override
    public GraphBuilder mutate(GraphBuilder builder, ComputationGraph prevGraph) {
        final Map<String, FeedForwardLayer> changedLayers = new HashMap<>(); // To be filled in
        mutationLayerSupplier.get().forEach(layerName -> updateNoutOfLayer(changedLayers, builder, prevGraph, layerName));
        return builder;
    }

    private GraphBuilder updateNoutOfLayer(
            Map<String, FeedForwardLayer> changedLayers,
            GraphBuilder builder,
            ComputationGraph prevGraph,
            String layerName) {
        final FeedForwardLayer newLayerConf = changedLayers.computeIfAbsent(layerName,
                key -> (FeedForwardLayer) prevGraph.getLayer(layerName).conf().getLayer().clone());
        newLayerConf.setNOut(mutationFunction.applyAsInt((int) newLayerConf.getNOut()));

        final List<String> inputs = prevGraph.getConfiguration().getVertexInputs().get(layerName);
        builder.removeVertexKeepConnections(layerName)
                .addLayer(layerName, newLayerConf, inputs.toArray(new String[0]));

        updateNinOfOutputs(changedLayers, builder, prevGraph, layerName, prevGraph.layerSize(layerName) - newLayerConf.getNOut());
        return builder;
    }

    private void updateNinOfOutputs(
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
                        newLayerConf.setNIn(newLayerConf.getNIn() - nNinDelta);
                        final List<String> vertexInputs = prevGraph.getConfiguration().getVertexInputs().get(newLayerConf.getLayerName());
                        builder.removeVertexKeepConnections(newLayerConf.getLayerName());

                        builder.addLayer(
                                newLayerConf.getLayerName(),
                                newLayerConf,
                                vertexInputs.toArray(new String[0]));
                    }
                    if (Mutation.doesNinPropagateToNext(graphVertex)) {
                        updateNinOfOutputs(changedLayers, builder, prevGraph, graphVertex.getVertexName(), nNinDelta);
                    }
                });
    }

}
