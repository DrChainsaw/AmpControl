package ampcontrol.model.training;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.listen.ActivationContribution;
import ampcontrol.model.training.model.EvolvingGraphAdapter;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelAdapter;
import ampcontrol.model.training.model.evolve.mutate.AggMutation;
import ampcontrol.model.training.model.evolve.mutate.NoutMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.BlockMutationFunction;
import ampcontrol.model.training.model.evolve.mutate.layer.GraphMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.LayerContainedMutation;
import ampcontrol.model.training.model.evolve.mutate.layer.LayerMutationInfo;
import ampcontrol.model.training.model.evolve.mutate.state.NoMutationStateWapper;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormAfter;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * Evaluates a model. Main use case is to test effects of various
 * {@link ampcontrol.model.training.model.evolve.mutate.Mutation}s.
 *
 * @author Christian Skärby
 */
class ModelEvaluationWorkBench {

    private static final Logger log = LoggerFactory.getLogger(TrainingDescription.class);

    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;

    ModelEvaluationWorkBench(
            MiniEpochDataSetIterator trainIter,
            MiniEpochDataSetIterator evalIter) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
    }

    public void evalute(final ComputationGraph graph, final String layer) {

         evalBaseline(graph);

        evalAddConvBlock(graph, layer);

        evalIncreaseNout(graph, layer);

        evalDecreaseNout(graph, layer);

        evalDecreaseNoutOptimal(graph, layer);

        evalIncreaseKernelSize(graph, layer);

        evalDecreaseKernelSize(graph, layer);
    }

    private void evalAddConvBlock(ComputationGraph graph, String layer) {
        final ModelAdapter addConvBlock = new EvolvingGraphAdapter(graph, new NoMutationStateWapper<>(
                AggMutation.<ComputationGraphConfiguration.GraphBuilder>builder()
                        .first(gb -> {
                            gb.setInputTypes(InputType.inferInputType(trainIter.next().getFeatures()));
                            return gb;
                        })
                        .andThen(new GraphMutation(() ->
                                Stream.of(GraphMutation.GraphMutationDescription.builder()
                                        .mutation(
                                                new BlockMutationFunction(
                                                        nOut ->
                                                                //new ResBlock().setBlockConfig(
                                                                        new Conv2DBatchNormAfter()
                                                                                .setConvolutionMode(ConvolutionMode.Same)
                                                                                .setKernelSize(3)
                                                                                .setNrofKernels(nOut.intValue())
                                                ,
                                                        new String[]{layer},
                                                        str -> "mut_" + str))
                                        .build())
                        ))
                        .build())).evolve();
        final ComputationGraph newGraph = (ComputationGraph) addConvBlock.asModel();
        final INDArray weigths = newGraph.getLayer("mut_0").paramTable().get(DefaultParamInitializer.WEIGHT_KEY);
        weigths.assign(Nd4j.zeros(weigths.shape()));
        for (int i = 0; i < weigths.size(0); i++) {
            weigths.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()},
                    Nd4j.create(new double[][]{
                            {0, 0, 0},
                            {0, 1, 0},
                            {0, 0, 0}}));
        }
        evaluateExperiment(addConvBlock, "addConvBlock: ");
    }


    private void evalDecreaseKernelSize(ComputationGraph graph, String layer) {
        final ModelAdapter decreaseKs = new EvolvingGraphAdapter(graph, new NoMutationStateWapper<>(
                new LayerContainedMutation(() ->
                        Stream.of(LayerContainedMutation.LayerMutation.builder()
                                .mutationInfo(LayerMutationInfo.builder()
                                        .layerName(layer)
                                        .build())
                                .mutation(layerConfIn -> Optional.ofNullable(layerConfIn)
                                        .map(Layer::clone)
                                        .filter(layerConf -> layerConf instanceof ConvolutionLayer)
                                        .map(layerConf -> (ConvolutionLayer) layerConf)
                                        .map(convConf -> {
                                            convConf.setKernelSize(new int[]{convConf.getKernelSize()[0] - 1, convConf.getKernelSize()[1]});
                                            convConf.setWeightInit(WeightInit.DISTRIBUTION);
                                            convConf.setDist(new ConstantDistribution(0));
                                            return convConf;
                                        })
                                        .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer from " + layerConfIn)))
                                .build()))
        )).evolve();
        evaluateExperiment(decreaseKs, "decreaseKernelSize: ");
    }

    private void evalIncreaseKernelSize(ComputationGraph graph, String layer) {
        final ModelAdapter increaseKs = new EvolvingGraphAdapter(graph, new NoMutationStateWapper<>(
                new LayerContainedMutation(() ->
                        Stream.of(LayerContainedMutation.LayerMutation.builder()
                                .mutationInfo(LayerMutationInfo.builder()
                                        .layerName(layer)
                                        .build())
                                .mutation(layerConfIn -> Optional.of(layerConfIn)
                                        .map(Layer::clone)
                                        .filter(layerConf -> layerConf instanceof ConvolutionLayer)
                                        .map(layerConf -> (ConvolutionLayer) layerConf)
                                        .map(convConf -> {
                                            convConf.setKernelSize(new int[]{convConf.getKernelSize()[0] + 1, convConf.getKernelSize()[1]});
                                            convConf.setWeightInit(WeightInit.DISTRIBUTION);
                                            convConf.setDist(new ConstantDistribution(0));
                                            return convConf;
                                        })
                                        .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer from " + layerConfIn)))
                                .build()))
        )).evolve();
        evaluateExperiment(increaseKs, "increaseKernelSize: ");
    }

    private void evalDecreaseNoutOptimal(ComputationGraph graph, String layer) {
        final Comparator<Integer> actContribComparator = getActivationContributionComparator(trainIter, layer, graph);
        final ModelAdapter decreaseNoutOpt = new EvolvingGraphAdapter(graph, new NoMutationStateWapper<>(
                new NoutMutation(() -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(layer)
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
        ),
                graphVar -> new ParameterTransfer(graphVar,
                        layerName -> Optional.of(i -> actContribComparator))).evolve();
        evaluateExperiment(decreaseNoutOpt, "decreaseNoutOpt: ");
    }

    private void evalDecreaseNout(ComputationGraph graph, String layer) {
        final ModelAdapter decreaseNout = new EvolvingGraphAdapter(graph, new NoMutationStateWapper<>(
                new NoutMutation(() -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(layer)
                                .mutateNout(nOut -> nOut - 1)
                                .build()))
        )).evolve();
        evaluateExperiment(decreaseNout, "decreaseNout: ");
    }

    private void evalIncreaseNout(ComputationGraph graph, String layer) {
        final ModelAdapter increaseNout = new EvolvingGraphAdapter(graph, new NoMutationStateWapper<>(
                new NoutMutation(() -> Stream.of(
                        NoutMutation.NoutMutationDescription.builder()
                                .layerName(layer)
                                .mutateNout(nOut -> nOut + 1)
                                .build()))
        )).evolve();
        evaluateExperiment(increaseNout, "increaseNout: ");
    }

    private void evalBaseline(ComputationGraph graph) {
        final ModelAdapter baseline = new GraphModelAdapter(graph);
        final String experimentLabel = "baseline: ";
        evaluateExperiment(baseline, experimentLabel);
    }

    private void evaluateExperiment(ModelAdapter modelAdapter, String experimentLabel) {
        Evaluation evaluation = new Evaluation(evalIter.getLabels());
        modelAdapter.eval(evalIter, evaluation);
        log.info(experimentLabel + evaluation.accuracy());
        evalIter.restartMiniEpoch();
    }

    private static Comparator<Integer> getActivationContributionComparator(
            MiniEpochDataSetIterator trainIter,
            String layer, ComputationGraph best) {

        final ComputationGraphConfiguration.GraphBuilder addEpsSpy = new ComputationGraphConfiguration.GraphBuilder(
                best.getConfiguration(),
                new NeuralNetConfiguration.Builder(best.conf())
        );
        final String epsSpyName = "EpsSpy_" + layer;
        new GraphMutation(() -> Stream.of(GraphMutation.GraphMutationDescription.builder()
                .mutation(gb -> {
                    gb.addVertex(epsSpyName, new EpsilonSpyVertex(), layer);
                    return GraphMutation.InputsAndOutputNames.builder()
                            .inputName(layer)
                            .outputName(epsSpyName)
                            .keepInputConnection(str -> str.equals(epsSpyName))
                            .build();
                })
                .build()))
                .mutate(addEpsSpy);

        class INDArrayCompsumer implements Comparator<Integer>, Consumer<INDArray> {

            private INDArray array;

            @Override
            public int compare(Integer o1, Integer o2) {
                return -Double.compare(array.getDouble(o1), array.getDouble(o2));
            }

            @Override
            public void accept(INDArray indArray) {
                if (array == null) {
                    array = indArray.detach();
                } else {
                    array.addi(indArray);
                }
            }
        }
        final INDArrayCompsumer actContribComparator = new INDArrayCompsumer();


        final ComputationGraph withEpsSpy = new ComputationGraph(addEpsSpy.build());
        withEpsSpy.init();
        new ParameterTransfer(best).transferWeightsTo(withEpsSpy);
        withEpsSpy.addListeners(new ActivationContribution("" + layer, arr -> {
            actContribComparator.accept(arr);
            log.debug("Contributions: " + arr);
        }));


        // Do a few training iterations to train the NoutMutation
        for (int i = 0; i < 3; i++) {
            withEpsSpy.fit(trainIter);
            trainIter.reset();
        }
        return actContribComparator;
    }
}
