package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description of a bunch of architectures which belong to a family of dense nets.
 * <br><br>
 * See https://arxiv.org/abs/1608.06993
 *
 * @author Christian Skärby
 */
public class DenseNetFactory {
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public DenseNetFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
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

        final int schedPeriod = 100;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.05));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));
        
        IntStream.of(2, 8).forEach(denseStackSize -> {
            Stream.of(new IdBlock(), new SeBlock()).forEach(seOrId -> {
                ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                        new BlockBuilder()
                                .setNamePrefix(namePrefix)
                                .setUpdater(new Nesterovs(lrSched, momSched))
                                .first(new ConvType(inputShape))
                                .andThen(new Conv2DBatchNormBetween()
                                        .setKernelSize(3)
                                        .setNrofKernels(64))
                                .andThen(new Pool2D().setSize(3).setStride(3))
                                .andThen(new Conv2DBatchNormBetween()
                                        .setKernelSize(3)
                                        .setConvolutionMode(ConvolutionMode.Same)
                                        .setNrofKernels(128))
                                .andThen(new Pool2D().setSize(3).setStride(3))
                                .andThen(seOrId)
                                .andThen(new Conv2DBatchNormBetween()
                                        .setKernelSize(3)
                                        .setConvolutionMode(ConvolutionMode.Same)
                                        .setNrofKernels(128))
                                .andThen(new Pool2D().setSize(3).setStride(3))
                                .andThen(seOrId)
                                .andThenStack(4)
                                .aggDenseStack(denseStackSize)
                                .aggOf(new Conv2DBatchNormBefore()
                                        .setKernelSize(3)
                                        .setConvolutionMode(ConvolutionMode.Same)
                                        .setNrofKernels(32))
                                .andFinally(new Conv2DBatchNormBefore()
                                        .setNrofKernels(32 * 2)
                                        .setKernelSize(1))
                                .andThen(new Conv2DBatchNormBefore()
                                        .setNrofKernels(32 * 4)
                                        .setKernelSize(1))
                                //   .andThen(new DropOut().setDropProb(dropOutProb))
                                .andFinally(seOrId)
                                .andThenStack(2)
                                .of(new Dense())//.setActivation(new ActivationSELU()))
                                .andFinally(new Output(trainIter.totalOutcomes())));
                modelData.add(new GenericModelHandle(
                        trainIter,
                        evalIter,
                        new GraphModelAdapter(builder.buildGraph()),
                        builder.name()));
            });
            });
    }
}
