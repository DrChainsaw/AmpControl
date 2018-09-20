package ampcontrol.model.training.model.evolve.mutate;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Inserts a layer in the graph. If a layer with the same name exists already it will be replaced. Note that for this
 * to work, the change of layer must be contained, i.e it must not "spill over" to other layers. Examples of what
 * "spills over" is changes in nIn or nOut as well as general changes in output shape in case layer is one of several
 * inputs to a vertex which require all their inputs to be of same shape.
 *
 * @author Christian Skärby
 */
public class MutateLayerContained implements Mutation<ComputationGraphConfiguration.GraphBuilder> {

    private static final Logger log = LoggerFactory.getLogger(MutateLayerContained.class);

    private final Supplier<LayerMutation> mutationSupplier;

    /**
     * Returns a {@link LayerVertex} with the given name if one exists in the {@link GraphBuilder}.
     * Would very much prefer to do this some other way, but the API does not seem allow for it
     */
    private static BiFunction<String, ComputationGraphConfiguration.GraphBuilder, Optional<LayerVertex>> vertexAsLayerVertex =
            (layerName, graphBuilder) -> Optional.ofNullable(graphBuilder.getVertices().get(layerName))
                    .filter(gv -> gv instanceof LayerVertex)
                    .map(gv -> ((LayerVertex) gv));

    /**
     * Returns a {@link FeedForwardLayer} of a given {@link LayerVertex} if that is the type of layer it encapsulates
     * Would very much prefer to do this some other way, but the API does not seem allow for it
     */
    private static BiFunction<String, Optional<LayerVertex>, Optional<FeedForwardLayer>> layerVertexAsFeedForward =
            (layerName, layerVertexOptional) -> layerVertexOptional
                    .map(LayerVertex::getLayerConf)
                    .map(NeuralNetConfiguration::getLayer)
                    .filter(layer -> layer instanceof FeedForwardLayer)
                    .map(layer -> (FeedForwardLayer) layer);

    public MutateLayerContained(Supplier<LayerMutation> mutationSupplier) {
        this.mutationSupplier = mutationSupplier;
    }

    @Builder
    @Getter
    public static class LayerMutation {
        private final String layerName;
        private final Function<Optional<Layer>, Layer> mutation;
        private final String[] inputLayers;

        private final BiFunction<String, GraphBuilder, Long> outputSizeMapping =
                (layerName, graphBuilder) -> vertexAsLayerVertex
                        .andThen(layerVertex -> layerVertexAsFeedForward.apply(layerName, layerVertex))
                        .apply(layerName, graphBuilder)
                        .map(FeedForwardLayer::getNOut)
                        .orElseGet(() -> graphBuilder.getVertexInputs().entrySet().stream()
                                .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(layerName))
                                .map(Map.Entry::getKey)
                                .mapToLong(inputLayerName -> getOutputSizeMapping().apply(inputLayerName, graphBuilder))
                                .sum());

        final BiFunction<String, GraphBuilder, Long> inputSizeMapping =
                (layerName, graphBuilder) -> vertexAsLayerVertex
                        .andThen(layerVertex -> layerVertexAsFeedForward.apply(layerName, layerVertex))
                        .apply(layerName, graphBuilder)
                        .map(FeedForwardLayer::getNIn)
                        .orElseGet(() -> Optional.ofNullable(graphBuilder.getVertexInputs().get(layerName))
                                .map(inputNames -> inputNames.stream()
                                        .mapToLong(inputLayerName -> getInputSizeMapping().apply(inputLayerName, graphBuilder))
                                        .sum())
                                .orElseThrow(() -> new IllegalStateException("No inputs found for " + layerName)));
    }


    @Override
    public GraphBuilder mutate(GraphBuilder builder) {
        mutationSupplier.stream().forEach(mutation -> replaceOrAddVertex(mutation, builder));
        return builder;
    }

    private void replaceOrAddVertex(LayerMutation mutation, GraphBuilder builder) {

        long nOut = 0;
        long nIn = 0;
        final Optional<Layer> toMutate;
        if (Optional.ofNullable(builder.getVertices().get(mutation.getLayerName())).isPresent()) {
            // Simple case, we can replace an existing vertex
            toMutate = Optional.of(((LayerVertex) builder.getVertices().get(mutation.getLayerName())).getLayerConf().getLayer());

            nOut = mutation.getOutputSizeMapping().apply(mutation.getLayerName(), builder);
            nIn = mutation.getInputSizeMapping().apply(mutation.getLayerName(), builder);
            builder.removeVertex(mutation.getLayerName(), false);
        } else {
            toMutate = Optional.empty();
            // Trickier case: Insert a vertex between two other vertices
            // vertexN -> vertexN+1 need to become vertexN -> newlayer -> vertexN+1
            // Thus, layerN+1 must be removed and added back with newLayer as input
            for (String inputName : mutation.getInputLayers()) {

                // Find all layers to which vertex of inputName is output
                final List<String> allOutputLayers = builder.getVertexInputs().entrySet().stream()
                        .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(inputName))
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList());

                for (String outputName : allOutputLayers) {

                    final GraphVertex vertexConf = builder.getVertices().get(outputName);

                    // Handle case of multiple inputs, like this:
                    // {vertex0,..., vertexN} -> outputVertex shall become vertexi -> newlayer and then {vertex0,...,newlayer,...,vertexN}-> outputVertex
                    // Must add back all other vertices as input to outputVertex.
                    final List<String> allInputsToOutput = new ArrayList<>(builder.getVertexInputs().get(outputName));
                    // Better insert new layer in the same place as the one which is removed...
                    final int index = allInputsToOutput.indexOf(inputName);
                    allInputsToOutput.remove(index);
                    allInputsToOutput.add(index, mutation.layerName);

                    // Assumes merge of inputs.
                    // Note: Will fail in case of a non-layer vertex. Needs fixing then...
                    nOut += mutation.getOutputSizeMapping().apply(outputName, builder);
                    nIn += mutation.getInputSizeMapping().apply(outputName, builder);

                    builder.removeVertex(outputName, false)
                            .addVertex(outputName,
                                    vertexConf,
                                    allInputsToOutput.toArray(new String[]{}));


                }
            }
        }

        final Layer mutatedLayer= mutation.getMutation().apply(toMutate);
        if (mutatedLayer instanceof FeedForwardLayer) {
            FeedForwardLayer ffLayer = ((FeedForwardLayer) mutatedLayer);
            ffLayer.setNIn(nIn);
            ffLayer.setNOut(nOut);
        }

        log.info("Mutated layer " + mutation.getLayerName() + " to " + mutatedLayer);

        builder.addLayer(mutation.getLayerName(), mutatedLayer, mutation.getInputLayers());
    }


}
