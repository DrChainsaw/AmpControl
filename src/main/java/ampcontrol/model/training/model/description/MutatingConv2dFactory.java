package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.listen.ActivationContribution;
import ampcontrol.model.training.listen.NanScoreWatcher;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.evolve.CachedPopulation;
import ampcontrol.model.training.model.evolve.EvolvingPopulation;
import ampcontrol.model.training.model.evolve.Population;
import ampcontrol.model.training.model.evolve.TransformPopulation;
import ampcontrol.model.training.model.evolve.fitness.AddListener;
import ampcontrol.model.training.model.evolve.fitness.AggPolicy;
import ampcontrol.model.training.model.evolve.fitness.ClearListeners;
import ampcontrol.model.training.model.evolve.fitness.FitnessPolicyTraining;
import ampcontrol.model.training.model.evolve.mutate.AggMutation;
import ampcontrol.model.training.model.evolve.mutate.MutateLayerContained;
import ampcontrol.model.training.model.evolve.mutate.MutateNout;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.selection.*;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphSpyAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.VertexSpyGraphAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import ampcontrol.model.training.model.naming.AddSuffix;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.apache.commons.lang.mutable.MutableLong;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.nd4j.jita.memory.CudaMemoryManager;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
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

    private final static class ActivationContributionComparator implements Consumer<INDArray>, Comparator<Integer> {

        private INDArray activationContribution = null;
        private final String wsName = "ActContribCompWs" + this.toString().split("@")[1];
        private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                //.policyMirroring(MirroringPolicy.HOST_ONLY)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policySpill(SpillPolicy.REALLOCATE)
                .initialSize(0)
                //.overallocationLimit(20)
                .build();

        @Override
        public int compare(Integer elem1, Integer elem2) {
            if(elem1.equals(elem2)) {
                return 0;
            }

            return -Double.compare(
                    activationContribution.getDouble(elem1),
                    activationContribution.getDouble(elem2));
        }

        @Override
        public void accept(INDArray activationContribution) {
           // log.info("Got contrib: " + activationContribution);
            try (MemoryWorkspace wss =  Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, wsName)) {
                if (this.activationContribution == null) {
                    this.activationContribution = activationContribution.dup().migrate(false);
                }
                this.activationContribution.addi(activationContribution);
            }
        }

        @Override
        protected void finalize() throws Throwable {
            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, wsName).destroyWorkspace();
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

        final List<EvolvingGraphAdapter> initialPopulation = new ArrayList<>();

        final Set<String> mutateNoutLayers = new LinkedHashSet<>();
        final GraphSpyAdapter.LayerSpy nOutSpy = (layerName, layer, layerInputs) -> {
            if (layer instanceof ConvolutionLayer) {
                mutateNoutLayers.add(layerName);
            } else if (layer instanceof DenseLayer) {
                mutateNoutLayers.add(layerName);
            }
        };

        final Set<MutateLayerContained.LayerMutation> mutateKernelSizeLayers = new LinkedHashSet<>();
        final GraphSpyAdapter.LayerSpy kernelSizeSpy = createKernelSizeSpy(mutateKernelSizeLayers);

        final ModelBuilder baseBuilder = createModelBuilder(lrSched, momSched, graphBuilderAdapter -> new VertexSpyGraphAdapter(
                new GraphSpyAdapter(
                        new GraphSpyAdapter(graphBuilderAdapter, kernelSizeSpy), nOutSpy),
                new EpsilonSpyVertex(), mutateNoutLayers::contains)
        );
        final Function<Integer, FileNamePolicy> modelNamePolicyFactory = candInd -> new AddSuffix(File.separator + candInd);
        final FileNamePolicy evolvingSuffix = new AddSuffix("_evolving_train");
        final ModelComparatorRegistry comparatorRegistry = new ModelComparatorRegistry();

        // Create model population
        IntStream.range(0, 20).forEach(candInd -> {

            final ModelBuilder builder = new DeserializingModelBuilder(
                    modelFileNamePolicy.compose(evolvingSuffix).andThen(modelNamePolicyFactory.apply(candInd)), baseBuilder);

            baseBuilder.buildGraph(); // Just to populate mutationLayers...
            final Mutation<ComputationGraphConfiguration.GraphBuilder> mutation =
                    AggMutation.<ComputationGraphConfiguration.GraphBuilder>builder()
                            .first(createNoutMutation(mutateNoutLayers, candInd))
                            .second(createKernelSizeMutation(mutateKernelSizeLayers, -candInd))
                            .build();

            final ComputationGraph graph = builder.buildGraph();
            log.info("Mutation layers: " + mutateNoutLayers);
            final EvolvingGraphAdapter adapter = candInd == 0 || graph.getIterationCount() > 0 ?
                    new EvolvingGraphAdapter(graph, mutation,
                            graphToTransfer -> new ParameterTransfer(graphToTransfer,
                                    Objects.requireNonNull(comparatorRegistry.get(graphToTransfer)))) :
                    new EvolvingGraphAdapter(mutateGraph(mutation, graph), mutation,
                            graphToTransfer -> new ParameterTransfer(graphToTransfer,
                                   Objects.requireNonNull(comparatorRegistry.get(graphToTransfer))));

            initialPopulation.add(adapter);
        });
        // Its either this or catch an exception since everything but the CudaMemoryManager throws an exception
        if(Nd4j.getMemoryManager() instanceof CudaMemoryManager) {
            Nd4j.getMemoryManager().purgeCaches();
        }

        final Random rng = new Random(666);
        final MutableLong nrofParams = new MutableLong(0);
        final Population<ModelHandle> population = new CachedPopulation<>(
                new TransformPopulation<>(adapter -> new GenericModelHandle(
                        trainIter,
                        evalIter,
                        adapter,
                        "cand"), // Do something about the name...
                        new EvolvingPopulation<>(
                                // The initial population
                                initialPopulation,

                                // Policy for computing fitness and as of now, do some cleanup and add some checks
                                // TODO: Separate out prepare and clean stuff from actual fitness policy or rename FitnessPolicy to CandidateCreationHook or something
                                AggPolicy.<EvolvingGraphAdapter>builder()
                                        // Not a fitness policy
                                        .first(new ClearListeners<>())
                                        // Kind of a fitness policy
                                        .second(new AddListener<>(fitnessConsumer -> new NanScoreWatcher(() -> fitnessConsumer.accept(Double.MAX_VALUE))))
                                        // Not a fitness policy
                                        .andThen((adapter, consumer) -> {
                                            for (String layerName : mutateNoutLayers) {
                                                final ActivationContributionComparator comparator = new ActivationContributionComparator();
                                                adapter.asModel().addListeners(new ActivationContribution(layerName, comparator));
                                                comparatorRegistry.add(adapter.asModel(), layerName, 0, comparator); // For Conv
                                                comparatorRegistry.add(adapter.asModel(), layerName, 1, comparator); // For Dense
                                            }
                                            return adapter;
                                        })
                                        // This is the actual fitness policy
                                        .andThen(new FitnessPolicyTraining<>(107))
                                        // Not a fitness policy
                                        .andThen((adapter, fitcons) -> {
                                            nrofParams.add(adapter.asModel().numParams());
                                            return adapter;
                                        })
                                        .build(),

                                // Polícy for selecting candidates after fitness has been reported

                                CompoundFixedSelection.<EvolvingGraphAdapter>builder()
                                        .andThen(2, new EliteSelection<>())
                                        .andThen(initialPopulation.size() -2,
                                                new EvolveSelection<>(
                                                        new RouletteSelection<EvolvingGraphAdapter>(rng::nextDouble)))
                                        .build()
                        )));

        population.onChangeCallback(() -> {
            log.info("Avg nrof params: " + (nrofParams.doubleValue() / initialPopulation.size()));
            nrofParams.setValue(0);
        });

        final FileNamePolicy referenceSuffix = new AddSuffix("_reference_train");
        modelData.add(new GenericModelHandle(trainIter, evalIter,
                new GraphModelAdapter(
                        new DeserializingModelBuilder(modelFileNamePolicy.compose(referenceSuffix),
                                baseBuilder).buildGraph()),
                referenceSuffix.toFileName(baseBuilder.name())));
        modelData.add(new ModelHandlePopulation(population, evolvingSuffix.toFileName(baseBuilder.name()), modelNamePolicyFactory));
    }

    @NotNull
    GraphSpyAdapter.LayerSpy createKernelSizeSpy(Set<MutateLayerContained.LayerMutation> mutateKernelSizeLayers) {
        final Random rng = new Random(666);
        final Set<String> layerNames = new HashSet<>();
        return (layerName, layer, layerInputs) -> {
            if (!layerNames.add(layerName)) {
                return;
            }

            if (layer instanceof ConvolutionLayer) {
                mutateKernelSizeLayers.add(MutateLayerContained.LayerMutation.builder()
                        .layerName(layerName)
                        .inputLayers(layerInputs)
                        .mutation(layerConfOpt -> layerConfOpt
                                .map(Layer::clone)
                                .filter(layerConf -> layerConf instanceof ConvolutionLayer)
                                .map(layerConf -> (ConvolutionLayer) layerConf)
                                .map(convConf -> {
                                    convConf.setKernelSize(IntStream.of(convConf.getKernelSize())
                                            .map(orgSize -> Math.min(10, Math.max(1, orgSize + 1 - rng.nextInt(3))))
                                            .toArray());
                                    return convConf;
                                })
                                .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer " + layerName + " from " + layerConfOpt)))
                        .build());
            }
        };
    }

    private Mutation<ComputationGraphConfiguration.GraphBuilder> createNoutMutation(final Set<String> mutationLayers, int seed) {
        final Random rng = new Random(seed);
        final Set<MutateNout.NoutMutation> nOutMutationSet = mutationLayers.stream()
                .map(str -> MutateNout.NoutMutation.builder()
                        .layerName(str)
                        .mutateNout(nOut -> (long) Math.max(4, nOut + Math.max(nOut, 10) * 0.5 * (rng.nextDouble() - 0.5)))
                        .build())
                .collect(Collectors.toSet());
        return new MutateNout(
                () -> nOutMutationSet.stream().filter(str -> rng.nextDouble() < 0.1));
    }

    private Mutation<ComputationGraphConfiguration.GraphBuilder> createKernelSizeMutation(
            final Set<MutateLayerContained.LayerMutation> mutationLayers,
            int seed) {
        final Random rng = new Random(seed);
        return new MutateLayerContained(
                () -> mutationLayers.stream().filter(str -> rng.nextDouble() < 0.1));
    }

    private ModelBuilder createModelBuilder(
            ISchedule lrSched,
            ISchedule momSched,
            UnaryOperator<GraphBuilderAdapter> spyFactory) {
        final LayerBlockConfig pool = new Pool2D().setSize(2).setStride(2);
        return new BlockBuilder()
                .setUpdater(new Nesterovs(lrSched, momSched))
                .setNamePrefix(namePrefix)
                .first(new ConvType(inputShape))
                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(32))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(64))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(128))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new GlobPool())
                .andThenStack(2)
                .of(new SpyBlock(new Dense()
                        .setHiddenWidth(128)
                        .setActivation(new ActivationReLU()))
                        .setFactory(spyFactory))
                .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                        .setAlpha(0.6)
                        .setLambda(0.0035));
    }

    @NotNull
    private static ComputationGraph mutateGraph(Mutation<ComputationGraphConfiguration.GraphBuilder> mutation, ComputationGraph graph) {
        final ComputationGraph mutated = new ComputationGraph(mutation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration().clone(),
                        new NeuralNetConfiguration.Builder(graph.conf().clone())))
                .build());
        mutated.init();
        return mutated;
    }

}
