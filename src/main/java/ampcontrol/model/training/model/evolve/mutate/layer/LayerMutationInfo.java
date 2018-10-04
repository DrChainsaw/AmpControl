package ampcontrol.model.training.model.evolve.mutate.layer;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Base class for mutating vertices. Apart from name itself, it contains input names and functions to calculate nIn and nOut
 * for a given layer in a given graph.
 *
 * @author Christian Skärby
 */
@Builder
@Getter
public class LayerMutationInfo {

    /**
     * Returns a {@link LayerVertex} with the given name if one exists in the {@link ComputationGraphConfiguration.GraphBuilder}.
     * Would very much prefer to do this some other way, but the API does not seem allow for it
     */
    private static final BiFunction<String, ComputationGraphConfiguration.GraphBuilder, Optional<LayerVertex>> vertexAsLayerVertex =
            (layerName, graphBuilder) -> Optional.ofNullable(graphBuilder.getVertices().get(layerName))
                    .filter(gv -> gv instanceof LayerVertex)
                    .map(gv -> ((LayerVertex) gv));

    /**
     * Returns a {@link FeedForwardLayer} of a given {@link LayerVertex} if that is the type of layer it encapsulates
     * Would very much prefer to do this some other way, but the API does not seem allow for it
     */
    private static final BiFunction<String, Optional<LayerVertex>, Optional<FeedForwardLayer>> layerVertexAsFeedForward =
            (layerName, layerVertexOptional) -> layerVertexOptional
                    .map(LayerVertex::getLayerConf)
                    .map(NeuralNetConfiguration::getLayer)
                    .filter(layer -> layer instanceof FeedForwardLayer)
                    .map(layer -> (FeedForwardLayer) layer);

    /**
     * Gets the output size of a named layer without building the graph. Will recurse up the graph until
     * a {@link FeedForwardLayer} is found in case given layer is not of type {@link FeedForwardLayer}. Assumption made:
     * Only {@link FeedForwardLayer}s have nIn != nOut.
     *
     * @param layerName    layer to determine output size for
     * @param graphBuilder Builder to search
     * @return the output size of the given layer
     */
    public static long getOutputSize(String layerName, ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        return findNextSize(layerName, graphBuilder, FeedForwardLayer::getNOut);
    }

    private static long findNextSize(
            String layerName,
            ComputationGraphConfiguration.GraphBuilder graphBuilder,
            Function<FeedForwardLayer, Long> sizeMapping) {
        return vertexAsLayerVertex
                .andThen(layerVertex -> layerVertexAsFeedForward.apply(layerName, layerVertex))
                .apply(layerName, graphBuilder)
                .map(sizeMapping)
                .orElseGet(() -> graphBuilder.getVertexInputs().entrySet().stream()
                        .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(layerName))
                        .map(Map.Entry::getKey)
                        .mapToLong(inputLayerName -> findNextSize(inputLayerName, graphBuilder, FeedForwardLayer::getNIn))
                        .findAny()
                        .orElseThrow(() -> new IllegalStateException("Could not find any feedforward layers after " + layerName)));
    }

    /**
     * Gets the input size of a named layer without building the graph. Will recurse down the graph until
     * and sum up nOut of all found {@link FeedForwardLayer} in case given layer is not of type {@link FeedForwardLayer}.
     * Assumption made: Only {@link FeedForwardLayer}s have nIn != nOut.
     *
     * @param layerName    layer to determine input size for
     * @param graphBuilder Builder to search
     * @return the input size of the given layer
     */
    public static long getInputSize(String layerName, ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        return sumPrevSize(layerName, graphBuilder, FeedForwardLayer::getNIn);
    }

    private static long sumPrevSize(String layerName,
                                    ComputationGraphConfiguration.GraphBuilder graphBuilder,
                                    Function<FeedForwardLayer, Long> sizeMapping) {
        return vertexAsLayerVertex
                .andThen(layerVertex -> layerVertexAsFeedForward.apply(layerName, layerVertex))
                .apply(layerName, graphBuilder)
                .map(sizeMapping)
                .orElseGet(() -> Optional.ofNullable(graphBuilder.getVertexInputs().get(layerName))
                        .map(inputNames -> inputNames.stream()
                                .mapToLong(inputLayerName -> sumPrevSize(inputLayerName, graphBuilder, FeedForwardLayer::getNOut))
                                .sum())
                        .orElseThrow(() -> new IllegalStateException("No inputs found for " + layerName)));

    }


    private final String layerName;
    private final String[] inputLayers;

    private final BiFunction<String, ComputationGraphConfiguration.GraphBuilder, Long> outputSizeMapping =
            LayerMutationInfo::getOutputSize;

    final BiFunction<String, ComputationGraphConfiguration.GraphBuilder, Long> inputSizeMapping =
            LayerMutationInfo::getInputSize;
}
