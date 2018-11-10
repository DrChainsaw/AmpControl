package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.util.*;
import org.apache.commons.lang.mutable.MutableLong;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
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
 * {@link GraphBuilder}.
 *
 * @author Christian Skärby
 */
public class RemoveVertexFunction implements Function<GraphBuilder, GraphMutation.InputsAndOutputNames> {

    private static final Logger log = LoggerFactory.getLogger(RemoveVertexFunction.class);

    private final String vertexNameToRemove;

    public RemoveVertexFunction(String vertexNameToRemove) {
        this.vertexNameToRemove = vertexNameToRemove;
    }

    @Override
    public GraphMutation.InputsAndOutputNames apply(GraphBuilder graphBuilder) {

        // WTF is this about? graphBuilder.removeVertex(vertexName, true) will go through all vertexInputs and
        // remove vertexToRemove from the list of inputs. However, this list is typically created by Array.asList
        // which returns an immutable list. Here we replace that list with a mutable instance.
        graphBuilder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertexNameToRemove))
                .forEach(entry -> graphBuilder.getVertexInputs().put(entry.getKey(), new ArrayList<>(entry.getValue())));

        final List<String> outputNames = new ForwardOf(graphBuilder).children(vertexNameToRemove)
                .collect(Collectors.toList());

        final List<String> inputNames = new ArrayList<>(graphBuilder.getVertexInputs().get(vertexNameToRemove));

        final long nOut = Math.max(GraphBuilderUtil.getOutputSize(vertexNameToRemove, graphBuilder),
                new Connect<>(
                        TraverseBuilder.forwards(graphBuilder).build(),
                        TraverseBuilder.backwards(graphBuilder)
                                .enterCondition(vertex -> graphBuilder.getVertices().get(vertex) instanceof MergeVertex)
                                .traverseCondition(vertex -> !GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent())
                                .build()
                ).children(vertexNameToRemove)
                        .map(GraphBuilderUtil.asFeedforwardLayer(graphBuilder))
                        .filter(Optional::isPresent)
                        .map(Optional::get)
                        .mapToLong(FeedForwardLayer::getNOut)
                        .sum());
        final long nIn = GraphBuilderUtil.getInputSize(vertexNameToRemove, graphBuilder);
        log.info("Remove " + vertexNameToRemove + " with inputs " + inputNames + " and outputs " + outputNames +
                " nIn: " + nIn + " nOut: " + nOut);

        //System.out.println("Remove " + vertexNameToRemove + " with inputs " + inputNames + " and outputs " + outputNames +
        //        " nIn: " + nIn + " nOut: " + nOut);

        final Collection<String> connectedMergeVertices = handleMergeVertexOutputs(graphBuilder, outputNames);

        //System.out.println("after merge handling: " + outputNames);

        removeOrphanedElemWiseVertices(graphBuilder, outputNames);

        //System.out.println("after elemwise handling: " + outputNames + " input " + inputNames);

        final Map<String, Set<String>> inputNamesPerOutput = getInputNamesPerOutput(graphBuilder, outputNames, inputNames);

        //System.out.println("input per output: " + inputNamesPerOutput);

        outputNames.stream()
                .peek(name -> log.info("Connect " + name + " to new inputs: " + inputNamesPerOutput.get(name)))
                //.peek(name -> //System.out.println("Connect " + name + " to new inputs: " + inputNamesPerOutput.get(name)))
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

            //System.out.println("change nout : " + nOut);
            changeNoutOfInputs(graphBuilder, inputNames, nOut);

            // Need to update other layers which have one of inputNames as their inputs
            inputNames.stream()
                    .map(name -> new ForwardOf(graphBuilder).children(name).collect(Collectors.toList()))
                    .forEach(names -> changeNinOfOutputs(
                            graphBuilder,
                            names,
                            nOut));

            changeNoutOfInputs(
                    graphBuilder,
                    connectedMergeVertices,
                    nOut);
        }

        return GraphMutation.InputsAndOutputNames.builder().build();
    }

    @NotNull
    private Map<String, Set<String>> getInputNamesPerOutput(GraphBuilder graphBuilder, List<String> outputNames, List<String> inputNames) {
        return outputNames.stream()
                .map(name -> new AbstractMap.SimpleEntry<>(name, new LinkedHashSet<>(inputNames)))
                .peek(entry -> entry.getValue().addAll(Optional.ofNullable(graphBuilder.getVertexInputs().get(entry.getKey())).orElse(new ArrayList<>())))
                .collect(Collectors.toMap(
                        AbstractMap.SimpleEntry::getKey,
                        AbstractMap.SimpleEntry::getValue
                ));
    }

    private void removeOrphanedElemWiseVertices(GraphBuilder builder, List<String> outputNames) {
        new ArrayList<>(outputNames).forEach(name -> {
            final GraphVertex vertex = builder.getVertices().get(name);
            if (vertex instanceof ElementWiseVertex && builder.getVertexInputs().get(name).size() <= 2) {
                // This seems like it could break for so many different reasons...
                // Other options are:
                // 1) Remove vertex in this function, its inputs and outputs needs to be kept track of so that they
                // can be joined. Also, their nOut and nIn is likely different compared to vertexNameToRemove
                // 2) Try to keep elementwise vertices and just give them new inputs. I can't see how this can
                // work given that vertexToRemove might just as well have been connected to a subsamplinglayer
                // -> need to remove it too if that is the case. Still, this could ripple all the way back to input
                //new BackwardOf(builder).children(name).forEach(outputNames::add);
                final Graph<String> forward = new ForwardOf(builder);
                final Graph<String> backward = new BackwardOf(builder);
                final Graph<String> findEndpoint = new Traverse<>(forward);
                final List<String> leafVertices = backward.children(name).flatMap(findEndpoint::children)
                        .filter(child -> forward.children(child).count() == 0)
                        .filter(child -> !builder.getNetworkOutputs().contains(child))
                        .collect(Collectors.toList());

                // These nodes must be input to the elemwise vertex.
                final String leafVertex;
                if(leafVertices.size() == 1) {
                    leafVertex = leafVertices.get(0);
                } else if(leafVertices.size() > 1) {
                    leafVertex = "mv_" + String.join("_", leafVertices);
                    builder.addVertex(leafVertex, new MergeVertex(), leafVertices.toArray(new String[0]));
                } else {
                    leafVertex = null;
                }

                if(leafVertices.size() != 0) {
                    builder.removeVertex(name, false)
                            .addVertex(name, new ElementWiseVertex(ElementWiseVertex.Op.Add), leafVertex);
                }

                //System.out.println("output just before elem remove " + name + " leaves " + leafVertices);
                new RemoveVertexFunction(name).apply(builder);
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
     *
     * @param builder     GraphBuilder with the current configuration
     * @param outputNames List of output names for the vertex which shall be removed
     * @return Don't know yet...
     */
    private Collection<String> handleMergeVertexOutputs(GraphBuilder builder, List<String> outputNames) {
        // Use this somehow to prevent that size changes in case of an elementwise vertex somewhere down the line
        // Probably also need to traverse through layers for which nIn and nOut can not be changed independently...
        final List<String> viableOutputs = new ArrayList<>();
        // No vertex after vertexNameToRemove may be input to this a vertex which is connected to the input of vertexNameToRemove as this
        // will create a cycle in the graph
        final Set<String> forbiddenOutputs = new Traverse<>(new BackwardOf(builder)).children(vertexNameToRemove).collect(Collectors.toSet());

        // Figure out two things at once:
        // 1) are any of the outputs connected to a merge vertex
        // and
        // 2) What are the other inputs to that merge vertex if 1)
        //final Map<String, List<String>> inputsToConnectedMergeVertex = getInputsConnectedToMergeVertex(builder, outputName);
        final Map<String, Set<String>> outputsToConnectedMergeVertex = new HashMap<>();
        final Set<String> pathToMerge = new LinkedHashSet<>();

        final Graph<String> backwards = TraverseBuilder.backwards(builder)
                .enterCondition(vertex -> true)
                .visitCondition(vertex -> !forbiddenOutputs.contains(vertex))
                .build();

        final Set<String> currentPath = new LinkedHashSet<>();
        TraverseBuilder.forwards(builder)
                .traverseCondition(vertex -> !isSizeChangePossible(builder.getVertices().get(vertex)))
                .enterListener(currentPath::add)
                .visitListener(vertex -> {
                      //System.out.println("visit: " + vertex);
                    if (builder.getVertices().get(vertex) instanceof MergeVertex) {
                        outputsToConnectedMergeVertex.put(vertex,
                                new LinkedHashSet<>(backwards.children(vertex)
                                        .filter(vert -> isSizeChangePossible(builder.getVertices().get(vert)))
                                        .collect(Collectors.toSet())));
                        //System.out.println("currpath: " + currentPath);
                        currentPath.stream()
                                // add vertices which are either 1) not mergevertices and 2) mergevertices with only 1 input (the one which is about to be removed)
                                .filter(childvertex -> outputsToConnectedMergeVertex.getOrDefault(childvertex, Collections.emptySet()).size() <= 1)
                               // .peek(vert -> //System.out.println("add to path: " + vert))
                                .forEach(pathToMerge::add);

                    }
                })
                .leaveListener(currentPath::remove)
                .build().children(vertexNameToRemove).forEach(vertex -> {/* Ignore */});
        ////System.out.println();
        //System.out.println("outputsConnectedToMergeVertex map " + outputsToConnectedMergeVertex);
        //System.out.println("path to merge: " + pathToMerge);

        outputNames.removeAll(pathToMerge);
        pathToMerge.add(vertexNameToRemove);

        pathToMerge.forEach(vertex -> {
            // "Loneley" mergevertices will be part of pathToMerge, need to remove them
            outputsToConnectedMergeVertex.remove(vertex);
            builder.removeVertex(vertex, true);
        });

        for (Set<String> inputNames : outputsToConnectedMergeVertex.values()) {
            inputNames.stream()
                    .findFirst()
                    .ifPresent(viableOutputs::add);
        }


        // Somewhere here we also want to add mergeVertexOutputs as output to viableOutputs
        // and maybe change the size. Or return some object which describes this action?

        outputNames.addAll(viableOutputs);
        outputNames.removeAll(outputsToConnectedMergeVertex.keySet());
        return outputsToConnectedMergeVertex.keySet();
    }

    private static void changeNoutOfInputs(GraphBuilder graphBuilder, Collection<String> inputNames, long nOut) {

        //System.out.println("inputnames: " + inputNames);
        // What we want here is to traverse in topological order really. Just so happens to be so that inputNames
        // is always in reverse topological order since this is how it is construced?
        final List<String> names = new ArrayList<>(inputNames);
        Collections.reverse(names);
        ////System.out.println("reverse: " + names);

        final SizeVisitor sizeRegistry = new SizeVisitor(
                new Traverse<>(vertex -> !GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent(),
                        new BackwardOf(graphBuilder)),
                graphBuilder,
                nOut,
                (layerSize, size) -> Math.max(1, size));
        inputNames.forEach(vertex -> sizeRegistry.set(vertex, nOut));

        toLayerStream(
                TraverseBuilder.backwards(graphBuilder)
                        .enterCondition(vertex -> true)
                        .enterListener(sizeRegistry::visit)
                        .build(),
                graphBuilder,
                names)
                .peek(layer -> log.info("Change nOut of layer " + layer.getLayerName() + " from " + layer.getNIn() + " to " + sizeRegistry.getSize(layer.getLayerName())))
                .forEachOrdered(layer -> {
                    final long thisNout = sizeRegistry.getSize(layer.getLayerName());
                    //System.out.println("change nOut of vertex " + layer.getLayerName() + " from " + layer.getNOut() + " to " + thisNout);
                    layer.setNOut(thisNout);
                    if (!isSizeChangePossible(layer)) {
                        layer.setNIn(thisNout);
                    }
                });
    }

    private static void changeNinOfOutputs(GraphBuilder graphBuilder, Collection<String> outputNames, long nIn) {
        //System.out.println("output names: " + outputNames);

        final MutableLong encounteredMergeVerticesMultiplier = new MutableLong(1);
        final Map<String, Long> nInToUse = outputNames.stream()
                .collect(Collectors.toMap(
                        name -> name,
                        name -> nIn
                ));
        toLayerStream(
                TraverseBuilder.forwards(graphBuilder)
                        .enterCondition(vertex -> !GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent())
                        .enterListener(vertex -> {
                            if (graphBuilder.getVertices().get(vertex) instanceof MergeVertex) {
                                encounteredMergeVerticesMultiplier.increment();
                            }
                        })
                        .leaveListener(vertex -> {
                            if (graphBuilder.getVertices().get(vertex) instanceof MergeVertex) {
                                encounteredMergeVerticesMultiplier.decrement();
                            }
                        })
                        .visitListener(vertex -> nInToUse.put(vertex, nIn * encounteredMergeVerticesMultiplier.longValue()))
                        .build(),
                graphBuilder,
                outputNames)
                .peek(layer -> log.info("Change nIn of layer " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nInToUse.get(layer.getLayerName())))
                .forEachOrdered(layer -> {
                    final long thisNIn = nInToUse.get(layer.getLayerName());
                    //System.out.println("change nIn of vertex " + layer.getLayerName() + " from " + layer.getNOut() + " to " + thisNIn);
                    layer.setNIn(thisNIn);
                    if (!isSizeChangePossible(layer)) {
                        layer.setNOut(thisNIn);
                    }
                });
    }

    private static Stream<FeedForwardLayer> toLayerStream(
            Graph<String> graph,
            GraphBuilder graphBuilder,
            Collection<String> names) {
        return
                Stream.concat(names.stream(), names.stream().flatMap(graph::children))
                        .map(GraphBuilderUtil.asFeedforwardLayer(graphBuilder))
                        .filter(Optional::isPresent)
                        .map(Optional::get);
    }

    /**
     * Return true if the given layer supports nIn != nOut
     *
     * @param layer the layer to check
     * @return true if the given layer supports nIn != nOut
     */
    private static boolean isSizeChangePossible(FeedForwardLayer layer) {
        return layer instanceof ConvolutionLayer
                || layer instanceof DenseLayer
                || layer instanceof BaseRecurrentLayer
                || layer instanceof BaseOutputLayer;
    }

    /**
     * Return true if the given vertex supports nIn != nOut
     *
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
