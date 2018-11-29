package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.util.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil.isSizeChangePossible;

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

        outputNames.addAll(connectedMergeVertices);
        inputNames.addAll(connectedMergeVertices);
        new InputOutputAlign(graphBuilder, outputNames, inputNames, nOut, nIn).invoke();

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
}
