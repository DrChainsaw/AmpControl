package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.util.*;
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

        final Collection<String> connectedMergeVertices = new ArrayList<>();

        // Skip the below if removal is trivial as handleMergeVertexOutputs tends to mess up stuff around MergeVertices in
        // a way so that it becomes harder to remove parts of the fork.
        final boolean sizeChange = isSizeChangePossible(graphBuilder.getVertices().get(vertexNameToRemove))
                || graphBuilder.getVertices().get(vertexNameToRemove) instanceof ElementWiseVertex;

        final boolean wasMergeVertex = graphBuilder.getVertices().get(vertexNameToRemove) instanceof MergeVertex;
        if (sizeChange) {
            connectedMergeVertices.addAll(handleMergeVertexOutputs(graphBuilder, outputNames));

            //System.out.println("after merge handling: " + outputNames);

            removeOrphanedElemWiseVertices(graphBuilder, outputNames);

            //System.out.println("after elemwise handling: " + outputNames + " input " + inputNames);

            removeRedunantMergeVertices(graphBuilder, inputNames, outputNames);
        } else {
            removeVertex(graphBuilder, vertexNameToRemove);
        }
        // System.out.println("Size change: " + sizeChange);

        //System.out.println("after redundant merge handling: " + outputNames + " input " + inputNames);

        final Map<String, Set<String>> inputNamesPerOutput = getInputNamesPerOutput(graphBuilder, outputNames, inputNames);

        //System.out.println("input per output: " + inputNamesPerOutput);

        outputNames.stream()
                .peek(name -> log.info("Connect " + name + " to new inputs: " + inputNamesPerOutput.get(name)))
                //.peek(name -> System.out.println("Connect " + name + " to new inputs: " + inputNamesPerOutput.get(name)))
                .forEach(outputName ->
                        graphBuilder.addVertex(
                                outputName,
                                graphBuilder.getVertices().get(outputName),
                                inputNamesPerOutput.get(outputName).toArray(new String[1])));

        if (!sizeChange && !wasMergeVertex) {
            return GraphMutation.InputsAndOutputNames.builder().build();
        }

        // Not possible to change network inputs (e.g. image size)
        final boolean isAnyLayerInputNetworkInput = graphBuilder.getNetworkInputs().stream()
                .anyMatch(inputNames::contains);

        // Do the change which adds neurons rather than the one which removes them
        // What about if nIn == nOut? Can't do early return it seems as this is no guarantee
        // that the below is not needed. Example when it is not involve pooling layers
        // and merge vertices

        // Why is not NoutMutation used for this? Because NoutMutation actually does something quite different:
        // Firstly it assumes that the graph is consistent to begin with w.r.t nOut and nIns (not the case here).
        // Secondly, it only needs to propagate the nOut forwards to subsequent layers nIn, only skipping
        // backwards when it encounters an ElementWiseVertex. Here we need to go backwards and fix the nOuts of
        // previous layers which are changed either because they are to be connected with the removed layers
        // outputs or because one of the paths in a fork was just removed.
        if (nIn > nOut || isAnyLayerInputNetworkInput) {
            // System.out.println("change nIn " + nIn);
            setNinOfOutputsToNoutSize(graphBuilder, connectedMergeVertices);
            setNinOfOutputsToNoutSize(graphBuilder, outputNames);
        } else {
            //System.out.println("change nout : " + nOut);
            changeNoutOfInputs(graphBuilder, inputNames, nOut);

            //System.out.println("do merges: " + connectedMergeVertices);
            changeNoutOfInputs(
                    graphBuilder,
                    connectedMergeVertices,
                    nOut);
        }

        return GraphMutation.InputsAndOutputNames.builder().build();

    }

    private static void removeVertex(GraphBuilder graphBuilder, String vertexNameToRemove) {
        // WTF is this about? graphBuilder.removeVertex(vertexName, true) will go through all vertexInputs and
        // remove vertexToRemove from the list of inputs. However, this list is typically created by Array.asList
        // which returns an immutable list. Here we replace that list with a mutable instance.
        graphBuilder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertexNameToRemove))
                .forEach(entry -> graphBuilder.getVertexInputs().put(entry.getKey(), new ArrayList<>(entry.getValue())));
        graphBuilder.removeVertex(vertexNameToRemove, true);
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
                if (leafVertices.size() == 1) {
                    leafVertex = leafVertices.get(0);
                } else if (leafVertices.size() > 1) {
                    leafVertex = "mv_" + String.join("_", leafVertices);
                    builder.addVertex(leafVertex, new MergeVertex(), leafVertices.toArray(new String[0]));
                } else {
                    leafVertex = null;
                }

                if (leafVertices.size() != 0) {
                    builder.removeVertex(name, false)
                            .addVertex(name, new ElementWiseVertex(ElementWiseVertex.Op.Add), leafVertex);
                }

                //System.out.println("output just before elem remove " + name + " leaves " + leafVertices);
                new RemoveVertexFunction(name).apply(builder);
                outputNames.remove(name);
            }
        });
    }

    private static void removeRedunantMergeVertices(GraphBuilder builder, Collection<String> inputNames, Collection<String> outputNames) {

        final Graph<String> graph =
                new Filter<>(vertex -> builder.getVertices().get(vertex) instanceof MergeVertex,
                        new Filter<>(vertex -> builder.getVertexInputs().get(vertex).size() == 1,
                                //new Peek<>(vertex -> System.out.println("check redundant merge " + vertex),
                                TraverseBuilder.forwards(builder)
                                        .traverseCondition(vertex -> true) // could do better here, something like is an outputname or size change propagates
                                        .build()));

        inputNames.forEach(name -> graph.children(name).forEach(outputName -> {
                    if (outputNames.contains(outputName)) {
                        new ForwardOf(builder).children(outputName).forEach(outputNames::add);
                        //System.out.println("Remove redundant mergevertex: " + outputName);
                        outputNames.remove(outputName);
                    }
                    new RemoveVertexFunction(outputName).apply(builder);


                })
        );
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
        final Map<String, Set<String>> inputsToConnectedMergeVertex = new HashMap<>();
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
                    ///System.out.println("visit: " + vertex);
                    if (builder.getVertices().get(vertex) instanceof MergeVertex) {
                        inputsToConnectedMergeVertex.put(vertex,
                                new LinkedHashSet<>(backwards.children(vertex)
                                        .filter(vert -> isSizeChangePossible(builder.getVertices().get(vert)))
                                        .collect(Collectors.toSet())));
                        //System.out.println("currpath: " + currentPath);
                        currentPath.stream()
                                // add vertices which are either 1) not mergevertices and 2) mergevertices with only 1 input (the one which is about to be removed)
                                .filter(childvertex -> inputsToConnectedMergeVertex.getOrDefault(childvertex, Collections.emptySet()).size() <= 1)
                                //.peek(vert -> System.out.println("add to path: " + vert))
                                .forEach(pathToMerge::add);

                    }
                })
                .leaveListener(currentPath::remove)
                .build().children(vertexNameToRemove).forEach(vertex -> {/* Ignore */});
        //System.out.println("outputsConnectedToMergeVertex map " + inputsToConnectedMergeVertex);
        //System.out.println("path to merge: " + pathToMerge);

        outputNames.removeAll(pathToMerge);

        for (Set<String> inputNames : inputsToConnectedMergeVertex.values()) {
            inputNames.stream()
                    .filter(vertex -> !pathToMerge.contains(vertex))
                    .findFirst()
                    .ifPresent(viableOutputs::add);
        }

        outputNames.addAll(viableOutputs);
        outputNames.removeAll(inputsToConnectedMergeVertex.keySet());

        if (outputNames.isEmpty()) {
            log.info("Failed to remove vertex " + vertexNameToRemove + " with outputs to connected MergeVertex: " + inputsToConnectedMergeVertex);
            return Collections.emptyList();
        }
        pathToMerge.add(vertexNameToRemove);

        pathToMerge.forEach(vertex -> {
            // "Loneley" mergevertices will be part of pathToMerge, need to remove them
            //System.out.println("Remove " + vertex);
            inputsToConnectedMergeVertex.remove(vertex);
            removeVertex(builder, vertex);
        });

        // Whats going on here? We only need the "top" mergevertex when traversing downwards as we will visit all the
        // other vertices subsequently. This is required to split nOuts correctly through mergevertices
        // Maybe this belongs closer to nOut setting though...
        return inputsToConnectedMergeVertex.keySet().stream()
                .filter(vertex -> TraverseBuilder.forwards(builder).build().children(vertex)
                        .noneMatch(child -> inputsToConnectedMergeVertex.keySet().contains(child)))
                .collect(Collectors.toSet());
    }

    private static void changeNoutOfInputs(GraphBuilder graphBuilder, Collection<String> inputNames, long nOut) {

        // System.out.println("inputnames: " + inputNames);
        // What we want here is to traverse in topological order really. Just so happens to be so that inputNames
        // is always in reverse topological order since this is how it is constructed?
        final List<String> names = new ArrayList<>(inputNames);
        Collections.reverse(names);
        //System.out.println("reverse: " + names);

        final SizeVisitor sizeRegistry = createSizeVisitor(graphBuilder, nOut);
        inputNames.forEach(vertex -> sizeRegistry.set(vertex, nOut));

        final Set<String> changedLayers = new LinkedHashSet<>();
        toLayerStream(
                TraverseBuilder.backwards(graphBuilder)
                        .enterCondition(GraphBuilderUtil.changeSizePropagates(graphBuilder))
                        .enterListener(sizeRegistry::visit)
                        .build(),
                graphBuilder,
                names)
                .peek(layer -> log.info("Change nOut of layer " + layer.getLayerName() + " from " + layer.getNOut() + " to " + sizeRegistry.getSize(layer.getLayerName())))
                .forEachOrdered(layer -> {
                    final long thisNout = sizeRegistry.getSize(layer.getLayerName());
                    // System.out.println("change nOut of vertex " + layer.getLayerName() + " from " + layer.getNOut() + " to " + thisNout);
                    layer.setNOut(thisNout);
                    if (!isSizeChangePossible(layer)) {
                        layer.setNIn(thisNout);
                    } else {
                        changedLayers.add(layer.getLayerName());
                    }
                });

        // Set Nin of layers which have changed and are not part of inputNames
        final Set<String> needToChangeNin = changedLayers.stream()
                .flatMap(vertex -> new ForwardOf(graphBuilder).children(vertex))
                // We only want to process feedforward layers.
                .filter(vertex -> GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent())
                .collect(Collectors.toSet());
        //System.out.println("Change nIns after changing nOuts");
        setNinOfOutputsToNoutSize(graphBuilder, needToChangeNin);
    }

    @NotNull
    private static SizeVisitor createSizeVisitor(GraphBuilder graphBuilder, long nOut) {
        final Graph<String> backward = new BackwardOf(graphBuilder);
        final Graph<String> traverseMerges = Traverse.leaves(
                vert -> graphBuilder.getVertices().get(vert) instanceof MergeVertex, backward);

        return new SizeVisitor(
                // Whats going on here? First, we want to traverse through MergeVertices to give fair sharing
                // between the inputs to them given that they can have different sizes and be of different numbers.
                // However, in case of an ElementWiseVertex, we don't want to do this as this would set the inputs
                // to the inputs to the MergeVertex to the same nOut as the nOut of the ElementWiseVertex -> error!
                vertex -> (graphBuilder.getVertices().get(vertex) instanceof ElementWiseVertex) ? backward.children(vertex) : traverseMerges.children(vertex),
                graphBuilder,
                nOut,
                (layerSize, size) -> Math.max(1, size));
    }

    private static void setNinOfOutputsToNoutSize(GraphBuilder graphBuilder, Collection<String> outputNames) {
        //System.out.println("output names: " + outputNames);
        log.info("Set NIn of outputs " + outputNames);

        final Deque<Long> limits = new ArrayDeque<>();
        final Graph<String> traverseInputs = TraverseBuilder.backwards(graphBuilder)
                .enterCondition(vertex -> true)
                .enterListener(vertex -> {
                    if (graphBuilder.getVertices().get(vertex) instanceof ElementWiseVertex) {
                        limits.push(1L);
                    } else {
                        limits.push(Long.MAX_VALUE);
                    }
                })
                .leaveListener(vertex -> limits.pop())
                .limitTraverse(limits::peekFirst)
                .traverseCondition(vertex -> !GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex).isPresent())
                .allowRevisit()
                .build();
        toLayerStream(
                TraverseBuilder.forwards(graphBuilder)
                        .enterCondition(GraphBuilderUtil.changeSizePropagates(graphBuilder))
                        .build(),
                graphBuilder,
                outputNames)
                .forEachOrdered(layer -> {
                    final long nInToUse = traverseInputs.children(layer.getLayerName())
                            .distinct()
                            .mapToLong(vertex -> GraphBuilderUtil.asFeedforwardLayer(graphBuilder).apply(vertex)
                                    .map(FeedForwardLayer::getNOut)
                                    .orElseGet(() -> graphBuilder.getNetworkInputs().contains(vertex)
                                            ? graphBuilder.getNetworkInputTypes().get(graphBuilder.getNetworkInputs().indexOf(vertex)).getShape(false)[0]
                                            : 0L))
                            .sum();
                    log.info("Change nIn of layer " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nInToUse);
                    if (nInToUse == 0) {
                        throw new RuntimeException("0 size for " + layer.getLayerName());
                    }
                    //System.out.println("change nIn of vertex " + layer.getLayerName() + " from " + layer.getNIn() + " to " + nInToUse);
                    layer.setNIn(nInToUse);
                    if (!isSizeChangePossible(layer)) {
                        layer.setNOut(nInToUse);
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
