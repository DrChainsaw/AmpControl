package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.params.CenterLossParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

/**
 * Transfers parameters from one {@link ComputationGraph} to another. Shapes must not be identical.
 *
 * @author Christian Skärby
 */
public class ParameterTransfer {

   // private static final Logger log = LoggerFactory.getLogger(ParameterTransfer.class);

    private final ComputationGraph graph;
    private final Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory;

    public ParameterTransfer(ComputationGraph graph) {
        this(graph, str -> Optional.empty());
    }

    ParameterTransfer(ComputationGraph graph, Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory) {
        this.graph = graph;
        this.compFactory = compFactory;
    }

    /**
     * Mutate the current graph into a new {@link ComputationGraphConfiguration} by transferring parameters
     *
     * @param newGraph new computation graph for which weights shall be transferred
     * @return A {@link ComputationGraph} initialized with parameters from the existing graph
     */
    public ComputationGraph transferWeightsTo(ComputationGraph newGraph) {

        final TransferRegistry registry = new TransferRegistry();
        final int[] topologicalOrder = newGraph.topologicalSortOrder();
        final GraphVertex[] vertices = newGraph.getVertices();
        //set params from orig graph as necessary to new graph
        for (int vertexIndex : topologicalOrder) {
            transferParameters(registry, newGraph, vertices[vertexIndex].getVertexName());
        }
        registry.commit();
        return newGraph;
    }

    private Optional<GraphVertex> findLayerVertex(String name, ComputationGraph graph) {
        return Stream.of(graph.getVertices())
                .filter(GraphVertex::hasLayer)
                .filter(vertex -> vertex.getLayer().numParams() > 0)
                .filter(vertex -> name.equals(vertex.getVertexName()))
                .findAny();
    }

    private void transferParameters(TransferRegistry registry, ComputationGraph newGraph, String layerName) {
        Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
        Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

           // log.info("Transfer parameters from " + sourceVertex.getVertexName());

            final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
            final Map<String, INDArray> targetParams = targetVertex.paramTable(false);

            if (sourceVertex.getLayer().paramTable().containsKey(DefaultParamInitializer.WEIGHT_KEY)) {
                TransferTask.ListBuilder taskBuilder = initReshapeListBuilder(registry, layerName, sourceParams, targetParams);
                transferOutputParameters(registry, newGraph, targetVertex, taskBuilder);
                taskBuilder.build().execute();
            } else {
                transferAllParameters(registry, layerName, sourceParams, targetParams);
            }
        }
    }

    private TransferTask.ListBuilder initReshapeListBuilder(
            TransferRegistry registry,
            String layerName,
            Map<String, INDArray> sourceParams,
            Map<String, INDArray> targetParams) {

        // 1 for conv, 0 for dense ?? for recurrent
        final int inputDim = sourceParams.get(DefaultParamInitializer.WEIGHT_KEY).shape().length == 4 ? 1 : 0;
        final SingleTransferTask.Builder firstTaskBuilder = SingleTransferTask.builder()
                .maskDim(inputDim) // Shall be transferred based on cropped outputs from previous layer
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(sourceParams.get(DefaultParamInitializer.WEIGHT_KEY), layerName + "_source_W"))
                        .build())
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(targetParams.get(DefaultParamInitializer.WEIGHT_KEY), layerName + "_target_W"))
                        .build());

        compFactory.apply(layerName)
                .ifPresent(firstTaskBuilder::compFactory);

        return firstTaskBuilder
                .addDependentTask(SingleTransferTask.builder()
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(sourceParams.get(DefaultParamInitializer.BIAS_KEY), layerName + "_source_b"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : 1)
                                .build())
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(targetParams.get(DefaultParamInitializer.BIAS_KEY), layerName + "_target_b"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim)
                                .build())
                );
    }

    private void transferAllParameters(TransferRegistry registry,
                                       String layerName,
                                       Map<String, INDArray> sourceParams,
                                       Map<String, INDArray> targetParams) {

        // TODO Which key is most important?
        sourceParams.keySet().stream().sorted().findFirst().ifPresent(startKey -> {
            final SingleTransferTask.Builder firstTaskBuilder = SingleTransferTask.builder()
                    .source(SingleTransferTask.IndMapping.builder()
                            .entry(registry.register(sourceParams.get(startKey), layerName + "_source_" + startKey))
                            .build())
                    .target(SingleTransferTask.IndMapping.builder()
                            .entry(registry.register(targetParams.get(startKey), layerName + "_target_" + startKey))
                            .build());

            compFactory.apply(layerName)
                    .ifPresent(firstTaskBuilder::compFactory);
            sourceParams.keySet().stream().filter(par -> !par.equals(startKey)).forEach(parKey -> {
                firstTaskBuilder
                        .addDependentTask(SingleTransferTask.builder()
                                .source(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(sourceParams.get(parKey), layerName + "_source_" + parKey))
                                        .build())
                                .target(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(targetParams.get(parKey), layerName + "_target_" + parKey))
                                        .build()));
            });
            firstTaskBuilder.build().execute();
        });

    }

    private void transferOutputParameters(
            TransferRegistry registry,
            ComputationGraph newGraph,
            GraphVertex rootNode,
            TransferTask.ListBuilder taskListBuilder) {

        Stream.of(Optional.ofNullable(rootNode.getOutputVertices()).orElse(new VertexIndices[0]))
                .map(vertexIndex -> newGraph.getVertices()[vertexIndex.getVertexIndex()])
                .forEachOrdered(vertex -> {
                    final String layerName = vertex.getVertexName();

                    Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
                    Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

                    if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
                        final GraphVertex sourceVertex = sourceVertexMaybe.get();
                        final GraphVertex targetVertex = targetVertexMaybe.get();

                       // log.info("Transfer output parameters of " + layerName);
                        final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
                        final Map<String, INDArray> targetParams = targetVertex.paramTable(false);
                        addTasksFor(registry, taskListBuilder, layerName, sourceParams, targetParams);
                    }
                    if (Mutation.doesNinPropagateToNext(vertex)) {
                        transferOutputParameters(registry, newGraph, vertex, taskListBuilder);
                    }
                });
    }

    private void addTasksFor(TransferRegistry registry,
                             TransferTask.ListBuilder taskListBuilder,
                             String layerName,
                             Map<String, INDArray> sourceParams,
                             Map<String, INDArray> targetParams) {

        for (String parKey : sourceParams.keySet()) {
            outputToInputDimMapping(parKey, sourceParams.get(parKey).shape().length).ifPresent(dimMapper -> {

                final INDArray sourceParam = sourceParams.get(parKey);
                final INDArray targetParam = targetParams.get(parKey);

                taskListBuilder.addDependentTask(SingleTransferTask.builder()
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(sourceParam, layerName + "_source_" + parKey))
                                .dimensionMapper(dimMapper)
                                .build())
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(targetParam, layerName + "_target_" + parKey))
                                .dimensionMapper(dimMapper) // flip dim 0 and 1
                                .build()));
            });
        }
    }

    /**
     * Describes how to map output dimension from a previous layer to a input dimension of a next layer for different parameters
     *
     * @param paramName The weight key name
     * @return The mapping. empty means weight key shall not be mapped
     */
    private Optional<IntUnaryOperator> outputToInputDimMapping(String paramName, int rank) {
        switch (paramName) {
            case (DefaultParamInitializer.WEIGHT_KEY):
                switch (rank) {
                    case 2: return Optional.of(IntUnaryOperator.identity());
                    case 4: return Optional.of(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim);
                    default: throw new UnsupportedOperationException("Not supported yet: " + rank);
                }
            case(CenterLossParamInitializer.CENTER_KEY):
                return Optional.of(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim);
            case (DefaultParamInitializer.BIAS_KEY):
            case (BatchNormalizationParamInitializer.BETA):
            case (BatchNormalizationParamInitializer.GAMMA):
            case (BatchNormalizationParamInitializer.GLOBAL_MEAN):
            case (BatchNormalizationParamInitializer.GLOBAL_VAR):
                return Optional.empty();
            default:
                throw new UnsupportedOperationException("Param type not supported: " + paramName);
        }
    }
}
