package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.Evolving;
import ampcontrol.model.training.model.evolve.mutate.MutateNout;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;


/**
 * {@link ModelAdapter} which may evolve through a {@link Mutation}.
 *
 * @author Christian Skärby
 */
public class EvolvingGraphAdapter implements ModelAdapter, Evolving<EvolvingGraphAdapter> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingGraphAdapter.class);

    private final ComputationGraph graph;
    private final Mutation mutation;

    final MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(WorkspaceConfiguration.builder()
                    .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .overallocationLimit(1.2)
                    .policyLearning(LearningPolicy.OVER_TIME)
                    .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                    .policySpill(SpillPolicy.REALLOCATE)
                    .initialSize(0)
                    .build(),
            this.getClass().getSimpleName() + "Workspace" + this.toString().split("@")[1]);

    public EvolvingGraphAdapter(ComputationGraph graph, Mutation mutation) {
        this.graph = graph;
        this.mutation = mutation;
    }

    @Override
    public void fit(DataSetIterator iter) {
        graph.fit(iter);
    }

    @Override
    public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals) {
        return graph.doEvaluation(iter, evals);
    }

    @Override
    public Model asModel() {
        return graph;
    }

    /**
     * Evolve the graph adapter
     * @return the evolved adapter
     */
    @Override
    public synchronized EvolvingGraphAdapter evolve() {
        try(MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            log.info("Evolve " + this);
            graph.getListeners().clear();
            final TransferLearning.GraphBuilder mutated = mutation.mutate(new TransferLearning.GraphBuilder(graph), graph);
            final ParameterTransfer parameterTransfer = new ParameterTransfer(graph);
            final ComputationGraph newGraph = mutated.build();
            newGraph.getConfiguration().setIterationCount(graph.getIterationCount());
            newGraph.getConfiguration().setEpochCount(graph.getEpochCount());
            return new EvolvingGraphAdapter(parameterTransfer.transferWeightsTo(newGraph), mutation);
        }
    }


    public static void main(String[] args) {
        int cnt = 0;
        final String inputName = "input";
        final String outputName = "output";
        final String conv1Name = "conv1";
        final String conv2Name = "conv2";
        final String conv3Name = "conv3";
        final int conv1Nout = 64;
        ComputationGraph graph = getNewGraph(inputName, outputName, conv1Name, conv2Name, conv3Name, conv1Nout);


        final DataSet dummySet = new DataSet(Nd4j.randn(new long[] {1,3,100,100}), Nd4j.randn(new long[] {1,10}));
        final Random rng = new Random(568);
        final Random rng2 = new Random(566);
        EvolvingGraphAdapter adapter = new EvolvingGraphAdapter(graph, new MutateNout(() -> Stream.of(conv1Name),
                i -> 63)); //rng.nextInt(200)+1));
        List<EvolvingGraphAdapter> adapterList = Stream.generate(() -> adapter).map(a -> rng2.nextBoolean() ? a.evolve() : a)
                .limit(10)
                .peek(a -> System.out.println("got: " + a))
                .collect(Collectors.toList());
        while (true) {
            //newGraph = getTransferedGraph(conv1Name, conv2Name, conv1Nout, 128-cnt, graph);

            // This also leaks, but not as fast as the above...
            // graph = getNewGraph(inputName, outputName, conv1Name, conv2Name, conv1Nout-cnt);

            System.out.println(cnt + " npar : " + graph.numParams());
            // Comment out to stop memory leak (!?). Each created graph is identical to the previous
            cnt++;
            if(cnt == 120) {
                cnt = 0;
            }
            for(EvolvingGraphAdapter graphAdapter: adapterList) {
                final ComputationGraph gg = (ComputationGraph) graphAdapter.asModel();
                System.out.println(graphAdapter);
                for (int i = 0; i < 100; i++) {
                    gg.fit(dummySet);
                }
            }
            adapterList = adapterList.stream().map(a -> rng2.nextBoolean() ? a.evolve() : a)
                    .collect(Collectors.toList());
        }
    }



    static ComputationGraph getTransferedGraph(String inputName, String layerName,int layerIn, int layerOut, ComputationGraph graph) {

        ComputationGraph newGraph = //new TransferLearning.GraphBuilder(
                // new TransferLearning.GraphBuilder(
                new TransferLearning.GraphBuilder(graph)
                        .removeVertexKeepConnections(layerName)
                        .addLayer(layerName, new ConvolutionLayer.Builder(3,3)
                                .nIn(layerIn)
                                .nOut(layerOut)
                                .build(), inputName)
                        //.build())
                        .removeVertexKeepConnections(batchNormName)
                        .addLayer(batchNormName, new BatchNormalization.Builder()
                                .nOut(layerOut)
                                .nIn(layerOut)
                                .build(), layerName)
                        //  .build())
                        .removeVertexKeepConnections("conv3")
                        .addLayer("conv3", new Convolution2D.Builder(3, 3)
                                .nIn(layerOut)
                                .nOut(256)
                                .build(), batchNormName)

                        .build();
        return new ParameterTransfer(graph).transferWeightsTo(newGraph);
        // return graph;
    }

    private final static String inputName = "input";
    private final static String outputName = "output";
    private final static String poolName = "pool";
    private final static String batchNormName = "batchNorm";
    private final static String globPoolName = "globPool";
    private final static String denseName = "dense";
    public final static String W = DefaultParamInitializer.WEIGHT_KEY;
    public final static String B = DefaultParamInitializer.BIAS_KEY;
    @NotNull
    public static ComputationGraph getNewGraph(String inputName, String outputName, String conv1Name, String conv2Name,String conv3Name, int conv1Nout) {
        ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(100, 100, 3))
                .addLayer(conv1Name, new Convolution2D.Builder(3, 3)
                        .nOut(conv1Nout)
                        .build(), inputName)
                .addLayer(conv2Name, new Convolution2D.Builder(3, 3)
                        .nOut(128)
                        .build(), conv1Name)
                .addLayer(batchNormName, new BatchNormalization.Builder().build(), conv2Name)
                .addLayer(conv3Name, new Convolution2D.Builder(3, 3)
                        .nOut(256)
                        .build(), batchNormName)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), conv3Name)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(256)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(10)
                        .build(), denseName)
                .build());
        graph.init();
        return graph;
    }
}
