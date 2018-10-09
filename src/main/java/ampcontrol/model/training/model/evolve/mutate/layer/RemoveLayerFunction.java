package ampcontrol.model.training.model.evolve.mutate.layer;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Function to be used with a {@link GraphMutation}. Removes a named layer from the given
 * {@link ComputationGraphConfiguration.GraphBuilder}.
 *
 * @author Christian Skärby
 */
public class RemoveLayerFunction implements Function<ComputationGraphConfiguration.GraphBuilder, GraphMutation.InputsAndOutputNames> {

    private static final Logger log = LoggerFactory.getLogger(RemoveLayerFunction.class);

    private final String vertexNameToRemove;

    public RemoveLayerFunction(String vertexNameToRemove) {
        this.vertexNameToRemove = vertexNameToRemove;
    }

    @Override
    public GraphMutation.InputsAndOutputNames apply(ComputationGraphConfiguration.GraphBuilder graphBuilder) {

        final List<String> outputNames = graphBuilder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertexNameToRemove))
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        final List<String> inputNames = graphBuilder.getVertexInputs().get(vertexNameToRemove);

        final long nOut = LayerMutationInfo.getOutputSize(vertexNameToRemove, graphBuilder);
        final long nIn = LayerMutationInfo.getInputSize(vertexNameToRemove, graphBuilder);
        graphBuilder.removeVertex(vertexNameToRemove, true);

        outputNames.forEach(outputName ->
                graphBuilder.addVertex(
                        outputName,
                        graphBuilder.getVertices().get(outputName),
                        inputNames.toArray(new String[0])));

        log.info("Remove " + vertexNameToRemove + " with inputs " + inputNames + " and outputs " + outputNames + " nOut: " + nOut);

        // Not possible to change network inputs (e.g. image size)
        final boolean isAnyLayerInputNetworkInput = graphBuilder.getNetworkInputs().stream()
                .anyMatch(inputNames::contains);

        // Do the change which adds neurons rather than the one which removes them
        if (nIn > nOut || isAnyLayerInputNetworkInput) {
            outputNames.stream()
                    .map(layerName -> findNextFeedForwardLayer(layerName, graphBuilder, nOut))
                    .filter(Optional::isPresent)
                    .map(Optional::get)
                    .peek(layer -> log.info("Change nIn of layer " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nIn))
                    .forEach(layer -> layer.setNIn(nIn));
        } else {
            inputNames.stream()
                    .map(layerName -> findPrevFeedForwardLayer(layerName, graphBuilder, nOut))
                    .filter(Optional::isPresent)
                    .map(Optional::get)
                    .peek(layer -> log.info("Change nOut of layer " + layer.getLayerName() + " from " + layer.getNOut() + " to " + nOut))
                    .forEach(layer -> layer.setNOut(nOut));
        }

        return GraphMutation.InputsAndOutputNames.builder().build();
    }

    private static Optional<FeedForwardLayer> findPrevFeedForwardLayer(
            String layerName,
            ComputationGraphConfiguration.GraphBuilder graphBuilder,
            long newNout) {
        return LayerMutationInfo.vertexAsLayerVertex
                .andThen(layerVertex -> LayerMutationInfo.layerVertexAsFeedForward.apply(layerName, layerVertex))
                .apply(layerName, graphBuilder)
                .filter(layer -> isTransitionLayer(layer, newNout))
                .map(Optional::of)
                .orElseGet(() -> Optional.ofNullable(graphBuilder.getVertexInputs().get(layerName))
                        .flatMap(inputLayers -> inputLayers.stream()
                                .map(layerInputName -> findPrevFeedForwardLayer(layerInputName, graphBuilder, newNout))
                                .filter(Optional::isPresent)
                                .map(Optional::get)
                                .findAny()));
    }

    private static Optional<FeedForwardLayer> findNextFeedForwardLayer(
            String layerName,
            ComputationGraphConfiguration.GraphBuilder graphBuilder,
            long newNin) {
        return LayerMutationInfo.vertexAsLayerVertex
                .andThen(layerVertex -> LayerMutationInfo.layerVertexAsFeedForward.apply(layerName, layerVertex))
                .apply(layerName, graphBuilder)
                .filter(layer -> isTransitionLayer(layer, newNin))
                .map(Optional::of)
                .orElseGet(() -> graphBuilder.getVertexInputs().entrySet().stream()
                        .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(layerName))
                        .map(Map.Entry::getKey)
                        .map(layerInputName -> findNextFeedForwardLayer(layerInputName, graphBuilder, newNin))
                        .filter(Optional::isPresent)
                        .map(Optional::get)
                        .findAny());
    }

    private static boolean isTransitionLayer(FeedForwardLayer layer, long newNout) {
        if (layer instanceof ConvolutionLayer
                || layer instanceof DenseLayer
                || layer instanceof BaseRecurrentLayer
                || layer instanceof BaseOutputLayer) {
            return true;
        }
        layer.setNIn(newNout);
        layer.setNOut(newNout);
        return false;
    }
}
