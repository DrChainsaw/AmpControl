package ampcontrol.model.training.model.mutate;

import ampcontrol.model.training.model.mutate.reshape.ReshapeSubTaskList;
import ampcontrol.model.training.model.mutate.reshape.ReshapeTask;
import ampcontrol.model.training.model.mutate.reshape.SingleReshapeSubTask;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

public class MutationGraph {

    private final ComputationGraph graph;
    private final Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory;

    public MutationGraph(ComputationGraph graph) {
        this(graph, str -> Optional.empty());
    }

    public MutationGraph(ComputationGraph graph, Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory) {
        this.graph = graph;
        this.compFactory = compFactory;
    }

    /**
     * Mutate the current graph into a new {@link ComputationGraphConfiguration} by transferring parameters
     *
     * @param config configuration of the new graph
     * @return A {@link ComputationGraph} build from the given config initialized with parameters from the existing graph
     */
    public ComputationGraph mutateTo(ComputationGraphConfiguration config) {
        final ComputationGraphConfiguration orgConf = graph.getConfiguration();
        final ComputationGraph newGraph = new ComputationGraph(config);
        newGraph.init();

        final int[] topologicalOrder = newGraph.topologicalSortOrder();
        final GraphVertex[] vertices = newGraph.getVertices();
        final Set<String> transferredVertices = new LinkedHashSet<>();
        //set params from orig graph as necessary to new graph
        for (int i = 0; i < topologicalOrder.length; i++) {
            final String name = vertices[topologicalOrder[i]].getVertexName();
            if(!transferredVertices.contains(name)) {
                transferParameters(newGraph, vertices[topologicalOrder[i]].getVertexName(), transferredVertices);
            }
        }
        return newGraph;
    }

    private Optional<GraphVertex> findLayerVertex(String name, ComputationGraph graph) {
        return Stream.of(graph.getVertices())
                .filter(GraphVertex::hasLayer)
                .filter(vertex -> vertex.getLayer().numParams() > 0)
                .filter(vertex -> name.equals(vertex.getVertexName()))
                .findAny();
    }

    private void transferParameters(ComputationGraph newGraph, String layerName, Set<String> transferredVertices) {
        Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
        Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

            final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
            final Map<String, INDArray> targetParams = targetVertex.paramTable(false);

            ReshapeTask.Builder taskBuilder = ReshapeTask.builder()
                    .sourceShape(sourceParams.get(DefaultParamInitializer.WEIGHT_KEY).shape())
                    .targetShape(targetParams.get(DefaultParamInitializer.WEIGHT_KEY).shape());

            final SingleReshapeSubTask.Builder firstSubtaskBuilder = SingleReshapeSubTask.builder()
                    .source(sourceParams.get(DefaultParamInitializer.WEIGHT_KEY))
                    .target(targetParams.get(DefaultParamInitializer.WEIGHT_KEY));

            compFactory.apply(layerName)
                    .ifPresent(firstSubtaskBuilder::compFactory);

            ReshapeSubTaskList.Builder reshapeListBuilder = ReshapeSubTaskList.builder()
                    .instruction(firstSubtaskBuilder.build())
                    .instruction(SingleReshapeSubTask.builder()
                            .source(sourceParams.get(DefaultParamInitializer.BIAS_KEY))
                            .target(targetParams.get(DefaultParamInitializer.BIAS_KEY))
                            .sourceIndMapping(SingleReshapeSubTask.IndMapping.builder()
                                    .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : 1)
                                    .build())
                            .build());
            transferredVertices.add(layerName);

            Optional.ofNullable(targetVertex.getOutputVertices())
                    .ifPresent(vertexIndices -> {
                        for (VertexIndices vertexIndex : vertexIndices) {
                            final String name = newGraph.getVertices()[vertexIndex.getVertexIndex()].getVertexName();
                            transferredVertices.add(name);
                            transferOutputParameters(newGraph, name, reshapeListBuilder);
                        }
                    });

            taskBuilder.reshapeSubTask(reshapeListBuilder.build())
                    .build()
                    .reshape();

            sourceParams.entrySet();

        }

    }

    private void transferOutputParameters(
            ComputationGraph newGraph,
            String layerName,
            ReshapeSubTaskList.Builder sublistBuilder) {
        Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
        Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

            final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
            final Map<String, INDArray> targetParams = targetVertex.paramTable(false);
                final INDArray sourceParam = sourceParams.get(DefaultParamInitializer.WEIGHT_KEY);
                final INDArray targetParam = targetParams.get(DefaultParamInitializer.WEIGHT_KEY);

                sublistBuilder.instruction(SingleReshapeSubTask.builder()
                        .source(sourceParam)
                        .target(targetParam)
                        .sourceIndMapping(SingleReshapeSubTask.IndMapping.builder()
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim) // flip dim 0 and 1
                                .build())
                        .build());
        }

    }
}
