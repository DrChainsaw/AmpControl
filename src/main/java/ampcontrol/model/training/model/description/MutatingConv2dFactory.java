package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.iterators.WorkSpaceWrappingIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.evolve.EvolvingPopulation;
import ampcontrol.model.training.model.evolve.mutate.MutateNout;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.selection.CompoundFixedSelection;
import ampcontrol.model.training.model.evolve.selection.EliteSelection;
import ampcontrol.model.training.model.evolve.selection.EvolveSelection;
import ampcontrol.model.training.model.evolve.selection.RouletteSelection;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import ampcontrol.model.training.model.naming.AddSuffix;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.model.validation.EvalValidation;
import ampcontrol.model.training.model.validation.Skipping;
import ampcontrol.model.training.model.validation.Validation;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Description for simple 2D convolutional networs which will evolve during training to perform architecture search.
 *
 * @author Christian Skärby
 */
public final class MutatingConv2dFactory {

    private static final Logger log = LoggerFactory.getLogger(MutatingConv2dFactory.class);

    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;


    private static final class ModelFitness implements Validation.Factory<Evaluation> {

        private final Consumer<Double> fitnessConsumer;
        private final double paramScore;

        public ModelFitness(Consumer<Double> fitnessConsumer, int nrofParameters) {
            this.fitnessConsumer = fitnessConsumer;
            this.paramScore = Math.max(0, (1e8 - nrofParameters) / 1e10);

        }

        @Override
        public Validation<Evaluation> create(List<String> labels) {
            final Consumer<Evaluation> listener = eval -> fitnessConsumer.accept(calcFitness(eval));
            return new Skipping<>(dummy -> 20, 20,
                    new EvalValidation(
                            new Evaluation(labels),
                            listener
                    ));
        }

        private double calcFitness(Evaluation evaluation) {
            return 1d / (Math.round(evaluation.accuracy() * 100) / 100d + paramScore) - 1;
        }
    }

    public MutatingConv2dFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelFileNamePolicy = modelFileNamePolicy;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.05));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        final List<EvolvingGraphAdapter> population = new ArrayList<>();

        final Set<String> mutationLayers = new LinkedHashSet<>();
        final GraphSpyAdapter.LayerSpy spy = (layerName, layer, layerInputs) -> {
            if (layer instanceof ConvolutionLayer) {
                mutationLayers.add(layerName);
            } else if (layer instanceof DenseLayer) {
                // Doesn't work yet...
                // mutationLayers.add(layerName);
            }
        };

        final ModelBuilder baseBuilder = createModelBuilder(lrSched, momSched, spy);
        final Function<Integer, FileNamePolicy> modelNamePolicyFactory = candInd -> new AddSuffix(File.separator + candInd);
        final FileNamePolicy evolvingSuffix = new AddSuffix("_evolving");
        IntStream.range(0, 20).forEach(candInd -> {

            final ModelBuilder builder = new DeserializingModelBuilder(
                    modelFileNamePolicy.compose(evolvingSuffix).andThen(modelNamePolicyFactory.apply(candInd)), baseBuilder);

            final Mutation mutation = createMutation(mutationLayers, candInd);
            final ComputationGraph graph = builder.buildGraph();
            log.info("Mutation layers: " + mutationLayers);
            final EvolvingGraphAdapter adapter = candInd == 0 ?
                    new EvolvingGraphAdapter(graph, mutation) :
                    new EvolvingGraphAdapter(mutation.mutate(new TransferLearning.GraphBuilder(graph), graph).build(), mutation);

            population.add(adapter);
        });

        // Perhaps this all should be in some class instead of as a dangling reference...
        // Poltergeist warning? Although objects are still referenced and do their stuff, just that noone needs to store the explict reference.
        final List<ModelHandle> evolvingPopulation = new ArrayList<>();
        final MiniEpochDataSetIterator evolveIter = new WorkSpaceWrappingIterator(new CachingDataSetIterator(evalIter, 20));
        final Random rng = new Random(666);
        new EvolvingPopulation<>(
                population,
                (adapters, fitnessConsumer) -> {
                    evolvingPopulation.clear();
                    for (EvolvingGraphAdapter adapter : adapters) {
                        final ModelHandle handle = new GenericModelHandle(
                                trainIter,
                                evolveIter,
                                adapter,
                                "cand" + evolvingPopulation.size());
                        handle.registerValidation(new ModelFitness(fitness -> fitnessConsumer.accept(fitness, adapter), handle.getModel().numParams()));
                        evolvingPopulation.add(handle);
                    }
                },
                CompoundFixedSelection.<EvolvingGraphAdapter>builder()
                        .andThen(2, new EliteSelection<>())
                        .andThen(population.size() - 2,
                                new EvolveSelection<>(
                                        new RouletteSelection<EvolvingGraphAdapter>(rng::nextDouble)))
                        .build()
        ).initEvolvingPopulation();

        final FileNamePolicy referenceSuffix = new AddSuffix("_reference");
        modelData.add(new GenericModelHandle(trainIter, evalIter,
                new GraphModelAdapter(
                        new DeserializingModelBuilder(modelFileNamePolicy.compose(referenceSuffix),
                                baseBuilder).buildGraph()),
                referenceSuffix.toFileName(baseBuilder.name())));
        modelData.add(new ModelHandlePopulation(evolvingPopulation, evolvingSuffix.toFileName(baseBuilder.name()), modelNamePolicyFactory));
    }

    private Mutation createMutation(final Set<String> mutationLayers, int seed) {
        final Random rng = new Random(seed);
        return new MutateNout(
                //() -> Stream.empty(),
                () -> mutationLayers.stream().filter(str -> rng.nextDouble() < 0.3),
                nOut -> (int) Math.max(4, nOut + 16 * (rng.nextDouble() - 0.5)));
    }

    private ModelBuilder createModelBuilder(ISchedule lrSched, ISchedule momSched, GraphSpyAdapter.LayerSpy spy) {
        // TODO Deserialize all models
        final LayerBlockConfig pool = new Pool2D().setSize(2).setStride(2);
        return new BlockBuilder()
                .setUpdater(new Nesterovs(lrSched, momSched))
                .setNamePrefix(namePrefix)
                .first(new ConvType(inputShape))
                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(32), spy))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(64), spy))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(128), spy))
                .andThen(pool)

                .andThen(new GlobPool())
                .andThenStack(2)
                .of(new SpyBlock(new Dense()
                        .setHiddenWidth(128)
                        .setActivation(new ActivationReLU()), spy))
                .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                        .setAlpha(0.6)
                        .setLambda(0.004));
    }
}
