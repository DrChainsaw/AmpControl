package ampcontrol.model.training.model.evolve.mutate;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Mutation operation which changes nOut of layers
 *
 * @author Christian Skärby
 */
public class MutateNout implements Mutation<ComputationGraphConfiguration.GraphBuilder> {

    private static final Logger log = LoggerFactory.getLogger(MutateNout.class);

    private final Supplier<NoutMutation> mutationLayerSupplier;

    /**
     * Interface for mutating Nout
     */
    @Builder
    @Getter
    public static class NoutMutation {
        private final String layerName;
        private final UnaryOperator<Long> mutateNout;
    }


    public MutateNout(Supplier<NoutMutation> mutationLayerSupplier) {
        this.mutationLayerSupplier = mutationLayerSupplier;
    }

    @Override
    public GraphBuilder mutate(GraphBuilder builder) {
        final Map<String, FeedForwardLayer> changedLayers = new HashMap<>(); // To be filled in
        mutationLayerSupplier.stream().forEach(mutation -> updateNoutOfLayer(changedLayers, builder, mutation));
        return builder;
    }

    private GraphBuilder updateNoutOfLayer(
            Map<String, FeedForwardLayer> changedLayers,
            GraphBuilder builder,
            NoutMutation mutation) {
        final String layerName = mutation.getLayerName();


        final FeedForwardLayer newLayerConf = changedLayers.computeIfAbsent(layerName,
                key -> (FeedForwardLayer) ((LayerVertex) builder.getVertices().get(layerName)).getLayerConf().getLayer().clone());
        final long oldNout = newLayerConf.getNOut();
        newLayerConf.setNOut(mutation.getMutateNout().apply(oldNout));
        log.info("Mutating nOut of layer " + layerName + " from " + oldNout + " to " + newLayerConf.getNOut());
        final List<String> inputs = builder.getVertexInputs().get(layerName);
        builder.removeVertex(layerName, false)
                .addLayer(layerName, newLayerConf, inputs.toArray(new String[0]));

        updateNinOfOutputLayer(changedLayers, builder, layerName, oldNout - newLayerConf.getNOut());
        return builder;
    }

    private void updateNinOfOutputLayer(
            Map<String, FeedForwardLayer> changedLayers,
            GraphBuilder builder,
            String layerName,
            long nNinDelta) {
        final Map<String, List<String>> vertexInputss = new LinkedHashMap<>(builder.getVertexInputs());
        vertexInputss.entrySet().stream().filter(entry -> entry.getValue().contains(layerName))
                .map(Map.Entry::getKey)
                .forEachOrdered(outputName -> {
//        prevGraph.getVertexInputs().entrySet().stream().filter()
//        Stream.of(Optional.ofNullable(prevGraph.getVertex(layerName).getOutputVertices()).orElse(new VertexIndices[0]))
//                .map(vertexInd -> prevGraph.getVertices()[vertexInd.getVertexIndex()])
//                .filter(vertex -> !vertex.getVertexName().equals(layerName))
                    final GraphVertex graphVertex = builder.getVertices().get(outputName);
                    if (graphVertex instanceof LayerVertex && ((LayerVertex) graphVertex).getLayerConf().getLayer() instanceof FeedForwardLayer) { // Layer for which it is possible to set inputs


                        final FeedForwardLayer newLayerConf = changedLayers.computeIfAbsent(outputName,
                                key -> (FeedForwardLayer) ((LayerVertex) graphVertex).getLayerConf().getLayer().clone());

                        final long newNIn = newLayerConf.getNIn() - nNinDelta;
                        newLayerConf.setNIn(newNIn);
                        if (changeNinMeansChangeNout(newLayerConf)) {
                            newLayerConf.setNOut(newNIn);
                        }

                        final List<String> vertexInputs = builder.getVertexInputs().get(newLayerConf.getLayerName());
                        builder.removeVertex(newLayerConf.getLayerName(), false);

                        builder.addLayer(
                                newLayerConf.getLayerName(),
                                newLayerConf,
                                vertexInputs.toArray(new String[0]));
                    }
                    if (doesNinPropagateToNext(graphVertex)) {
                        updateNinOfOutputLayer(changedLayers, builder, outputName, nNinDelta);
                    }
                });
    }

    private static boolean changeNinMeansChangeNout(FeedForwardLayer layer) {

        // Is there any parameter which can tell this instead of hardcoding it to types like this?
        return layer instanceof BatchNormalization;
    }

    private static boolean doesNinPropagateToNext(GraphVertex vertex) {
        if(!(vertex instanceof LayerVertex)) {
            return true;
        }
        LayerVertex layerVertex = (LayerVertex)vertex;

        if(layerVertex.numParams(false) == 0) {
            return true;
        }



        if (layerVertex.getLayerConf().getLayer() instanceof BatchNormalization) {
            return true;
        }

        return false;
    }
}
