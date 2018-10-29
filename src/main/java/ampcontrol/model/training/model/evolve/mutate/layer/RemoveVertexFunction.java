package ampcontrol.model.training.model.evolve.mutate.layer;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Function to be used with a {@link GraphMutation}. Removes a named layer from the given
 * {@link ComputationGraphConfiguration.GraphBuilder}.
 *
 * @author Christian Skärby
 */
public class RemoveVertexFunction implements Function<ComputationGraphConfiguration.GraphBuilder, GraphMutation.InputsAndOutputNames> {

    private static final Logger log = LoggerFactory.getLogger(RemoveVertexFunction.class);

    private final String vertexNameToRemove;

    public RemoveVertexFunction(String vertexNameToRemove) {
        this.vertexNameToRemove = vertexNameToRemove;
    }

    @Override
    public GraphMutation.InputsAndOutputNames apply(ComputationGraphConfiguration.GraphBuilder graphBuilder) {

        // WTF is this about? graphBuilder.removeVertex(vertexName, true) will go through all vertexInputs and
        // remove vertexToRemove from the list of inputs. However, this list is typically created by Array.asList
        // which returns an immutable list. Here we replace that list with a mutable instance.
        graphBuilder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertexNameToRemove))
                .forEach(entry -> graphBuilder.getVertexInputs().put(entry.getKey(), new ArrayList<>(entry.getValue())));

        final List<String> outputNames = graphBuilder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertexNameToRemove))
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        final List<String> inputNames = new ArrayList<>(graphBuilder.getVertexInputs().get(vertexNameToRemove));

        final long nOut = LayerMutationInfo.getOutputSize(vertexNameToRemove, graphBuilder);
        final long nIn = LayerMutationInfo.getInputSize(vertexNameToRemove, graphBuilder);
        log.info("Remove " + vertexNameToRemove + " with inputs " + inputNames + " and outputs " + outputNames +
                " nIn: " + nIn + " nOut: " + nOut);

        System.out.println("Remove " + vertexNameToRemove + " with inputs " + inputNames + " and outputs " + outputNames +
                " nIn: " + nIn + " nOut: " + nOut);
        removeOrphanedElemWiseVertices(graphBuilder, outputNames);

        handleMergeVertexOutputs(graphBuilder, outputNames);

        final Map<String, List<String>> inputNamesPerOutput = getInputNamesPerOutput(graphBuilder, outputNames, inputNames);

        graphBuilder.removeVertex(vertexNameToRemove, true);

        outputNames.stream()
                .peek(name -> log.info("Connect " + name + " to " + inputNamesPerOutput.get(name)))
                .peek(name -> System.out.println("Connect " + name + " to " + inputNamesPerOutput.get(name)))
                .forEach(outputName ->
                graphBuilder.addVertex(
                        outputName,
                        graphBuilder.getVertices().get(outputName),
                        inputNamesPerOutput.get(outputName).toArray(new String[1])));

        // Not possible to change network inputs (e.g. image size)
        final boolean isAnyLayerInputNetworkInput = graphBuilder.getNetworkInputs().stream()
                .anyMatch(inputNames::contains);

        // Do the change which adds neurons rather than the one which removes them
        if (nIn > nOut || isAnyLayerInputNetworkInput) {
            changeNinOfOutputs(graphBuilder, outputNames, nIn);

        } else {
            changeNoutOfInputs(graphBuilder, inputNames, nOut);

            // Need to update other layers which have one of inputNames as their inputs
            inputNames.forEach(name -> changeNinOfOutputs(
                    graphBuilder,
                    graphBuilder.getVertexInputs().entrySet().stream()
                            .filter(entry -> entry.getValue().contains(name))
                            .filter(entry -> outputNames.stream().noneMatch(output -> entry.getKey().equals(output)))
                            .map(Map.Entry::getKey)
                            .collect(Collectors.toList()),
                    nOut));
        }

        return GraphMutation.InputsAndOutputNames.builder().build();
    }

    @NotNull
    private Map<String, List<String>> getInputNamesPerOutput(ComputationGraphConfiguration.GraphBuilder graphBuilder, List<String> outputNames, List<String> inputNames) {
        return outputNames.stream()
                .map(name -> new AbstractMap.SimpleEntry<>(name, new ArrayList<>(inputNames)))
                .peek(entry -> entry.getValue().addAll(graphBuilder.getVertexInputs().get(entry.getKey())))
                .peek(entry -> entry.getValue().remove(vertexNameToRemove))
                .collect(Collectors.toMap(
                        AbstractMap.SimpleEntry::getKey,
                        AbstractMap.SimpleEntry::getValue

                ));
    }

    private void removeOrphanedElemWiseVertices(ComputationGraphConfiguration.GraphBuilder builder, List<String> outputNames) {
        new ArrayList<>(outputNames).forEach(name -> {
            final GraphVertex vertex = builder.getVertices().get(name);
            if (vertex instanceof ElementWiseVertex && builder.getVertexInputs().get(name).size() <= 2) {

                outputNames.addAll(builder.getVertexInputs().entrySet().stream()
                        .filter(entry -> entry.getValue().contains(name))
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList()));
                builder.removeVertex(name, true);
                outputNames.remove(name);
            }
        });
    }

    /**
     * Handle the case when the vertex to remove is connected to a {@link MergeVertex} in such a way so that it is not
     * possible to change the input of the next vertex size without changing the output size of said {@link MergeVertex}.
     * The reason to avoid this is that one then could end up with the case where a layer A (which is input to the
     * vertex which shall be removed) is merged with other layers (which typically has A as input) and where A shall be
     * element wise added to the output of the {@link MergeVertex}. This is obviously not possible for any allowed size
     * of A or the other layers (since 0 is not an allowed size). This situation commonly occurs when using
     * residual inception modules.
     * <br><br>
     * Because of this, the adopted strategy is to "reroute" the inputs of the removed layer to the other layers
     * which are merged in the same {@link MergeVertex} while at the same time add the outputs of the removed layers
     * as outputs to one of said other layers. After nIn and nOut of other layers are updated with respect to the new
     * inputs and outputs, it should be guaranteed that the output of any subsequent {@link MergeVertex} remains
     * unchanged.
     * @param builder GraphBuilder with the current configuration
     * @param outputNames List of output names for the vertex which shall be removed
     * @return Don't know yet...
     */
    private List<String> handleMergeVertexOutputs(ComputationGraphConfiguration.GraphBuilder builder, List<String> outputNames) {
        // Use this somehow to prevent that size changes in case of an elementwise vertex somewhere down the line
        // Probably also need to traverse through layers for which nIn and nOut can not be changed independently...
        final List<String> mergeVertexOutputs = new ArrayList<>();
        final List<String> viableOutputs = new ArrayList<>();
        final Set<String> forbiddenOutputs = createForbiddenSet(builder, Collections.singleton(vertexNameToRemove));
        for (String outputName : outputNames) {
            // Figure out two things at once:
            // 1) is the output connected to a merge vertex
            // and
            // 2) What are the other inputs to that merge vertex if 1)
            final Map<String, List<String>> inputsToConnectedMergeVertex = getInputsConnectedToMergeVertex(builder, outputName);
            System.out.println("inputsConnected: " + inputsToConnectedMergeVertex);
            if (!inputsToConnectedMergeVertex.isEmpty()) {
                //outputName is connected to a MergeVertex
                mergeVertexOutputs.add(outputName);
            }
            for (List<String> inputNames : inputsToConnectedMergeVertex.values()) {
                inputNames.stream().flatMap(
                        name -> getNonSizeTransparentInputs(builder, forbiddenOutputs, name).stream())
                        .findFirst()
                        .ifPresent(viableOutputs::add);
            }
        }
        // Somewhere here we also want to add mergeVertexOutputs as output to viableOutputs
        // and maybe change the size. Or return some object which describes this action?

        System.out.println("viable outputs: " + viableOutputs);
        outputNames.removeAll(mergeVertexOutputs);
        outputNames.addAll(viableOutputs);
        return viableOutputs;
    }

    /**
     * Create the set of vertex names for which the given vertex may not be input to or else there will be a cycle in
     * the graph
     *
     * @param builder     represents the configuration
     * @param vertexNames Collections of the vertices
     * @return The forbidden set
     */
    private static Set<String> createForbiddenSet(ComputationGraphConfiguration.GraphBuilder builder, Collection<String> vertexNames) {
        final Set<String> forbidden = new HashSet<>();
        for (String vertexName : vertexNames) {
            if (!builder.getNetworkInputs().contains(vertexName)) {
                forbidden.addAll(builder.getVertexInputs().get(vertexName));
                forbidden.addAll(createForbiddenSet(builder, forbidden));
            }
        }
        return forbidden;
    }

    private static Map<String, List<String>> getInputsConnectedToMergeVertex(ComputationGraphConfiguration.GraphBuilder builder, String vertexName) {
        final GraphVertex vertex = builder.getVertices().get(vertexName);
        if (vertex instanceof MergeVertex) {
            return Collections.singletonMap(vertexName, new ArrayList<>(builder.getVertexInputs().get(vertexName)));
        }

        final Map<String, List<String>> inputsConnectedToMergeVertex = new LinkedHashMap<>();
        if (!isSizeChangePossible(vertex)) {
            List<String> outputNames = builder.getVertexInputs().entrySet().stream()
                    .filter(entry -> entry.getValue().contains(vertexName))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            for (String outputName : outputNames) {
                getInputsConnectedToMergeVertex(builder, outputName).forEach(
                        (mergeVertexName, inputs) -> inputsConnectedToMergeVertex.merge(mergeVertexName, inputs,
                                (inputs1, inputs2) -> Stream.concat(inputs1.stream(), inputs2.stream())
                                        .distinct()
                                        .collect(Collectors.toList()))
                );
            }
        }
        return inputsConnectedToMergeVertex;
    }

    private static List<String> getNonSizeTransparentInputs(
            ComputationGraphConfiguration.GraphBuilder builder,
            Set<String> forbidden,
            String vertexName) {

        if (!isSizeChangePossible(builder.getVertices().get(vertexName))) {
            return builder.getVertexInputs().get(vertexName).stream()
                    .flatMap(name -> getNonSizeTransparentInputs(builder, forbidden, name).stream())
                    .filter(name -> !forbidden.contains(name))
                    .collect(Collectors.toList());
        }
        return Collections.singletonList(vertexName);
    }

    private static void changeNoutOfInputs(ComputationGraphConfiguration.GraphBuilder graphBuilder, List<String> inputNames, long nOut) {
        inputNames.stream()
                .map(layerName -> findPrevFeedForwardLayer(layerName, graphBuilder, nOut))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .peek(layer -> log.info("Change nOut of layer " + layer.getLayerName() + " from " + layer.getNOut() + " to " + nOut))
                .forEach(layer -> layer.setNOut(nOut));
    }

    private static void changeNinOfOutputs(ComputationGraphConfiguration.GraphBuilder graphBuilder, List<String> outputNames, long nIn) {
        outputNames.stream()
                .map(layerName -> findNextFeedForwardLayer(layerName, graphBuilder, nIn))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .peek(layer -> log.info("Change nIn of layer " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nIn))
                .forEach(layer -> layer.setNIn(nIn));
    }


    private static Optional<FeedForwardLayer> findPrevFeedForwardLayer(
            String layerName,
            ComputationGraphConfiguration.GraphBuilder graphBuilder,
            long newNout) {
        return LayerMutationInfo.vertexAsLayerVertex
                .andThen(layerVertex -> LayerMutationInfo.layerVertexAsFeedForward.apply(layerName, layerVertex))
                .apply(layerName, graphBuilder)
                .filter(layer -> isSizeChangePossibleOrElseChange(layer, newNout))
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
                .filter(layer -> isSizeChangePossibleOrElseChange(layer, newNin))
                .map(Optional::of)
                .orElseGet(() -> graphBuilder.getVertexInputs().entrySet().stream()
                        .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(layerName))
                        .map(Map.Entry::getKey)
                        .map(layerInputName -> findNextFeedForwardLayer(layerInputName, graphBuilder, newNin))
                        .filter(Optional::isPresent)
                        .map(Optional::get)
                        .findAny());
    }


    /**
     * Return true if the given layer supports nIn != nOut. If not true, the size will also be changed of the given layer.
     * @param layer Layer to check
     * @param newNout New nOut (and nIn) to set in case size change not possible
     * @return true if the given layer supports nIn != nOut.
     */
    private static boolean isSizeChangePossibleOrElseChange(FeedForwardLayer layer, long newNout) {
        if (isSizeChangePossible(layer)) return true;
        layer.setNIn(newNout);
        layer.setNOut(newNout);
        return false;
    }

    /**
     * Return true if the given layer supports nIn != nOut
     * @param layer the layer to check
     * @return true if the given layer supports nIn != nOut
     */
    private static boolean isSizeChangePossible(FeedForwardLayer layer) {
        if (layer instanceof ConvolutionLayer
                || layer instanceof DenseLayer
                || layer instanceof BaseRecurrentLayer
                || layer instanceof BaseOutputLayer) {
            return true;
        }
        return false;
    }

    /**
     * Return true if the given vertex supports nIn != nOut
     * @param vertex the vertex to check
     * @return true if the given vertex supports nIn != nOut
     */
    private static boolean isSizeChangePossible(GraphVertex vertex) {
        if (vertex instanceof LayerVertex) {
            Layer layer = ((LayerVertex) vertex).getLayerConf().getLayer();
            if (layer instanceof FeedForwardLayer) {
                return isSizeChangePossible((FeedForwardLayer) layer);
            }
        }
        return false;
    }
}
