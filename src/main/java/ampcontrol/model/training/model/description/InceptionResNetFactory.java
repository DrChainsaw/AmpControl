package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Description of some homebrewed inception resnets with 2D convolutions
 *
 * @author Christian Skärby
 */
public class InceptionResNetFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public InceptionResNetFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelDir = modelDir;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {

        final LayerBlockConfig pool = new Pool2D().setSize(3).setStride(3); final int resblockOutFac = 1;
        // final LayerBlockConfig pool = new MinMaxPool().setSize(3).setStride(3); final int resblockOutFac = 2;

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new Step(4000, new Exponential(0.2)),
                new SawTooth(schedPeriod, 1e-6, 0.1));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        final int resNrofChannels = 64;
        IntStream.of(5).forEach(resDepth ->
                DoubleStream.of(0).forEach(dropOutProb ->
                        DoubleStream.of(0.004).forEach(lambda -> {
                            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                                    createStem(pool, lrSched, momSched, resNrofChannels)
                                            .andThenStack(resDepth)
                                            .res()
                                            .aggFork()
                                            .add(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(resNrofChannels)
                                            .setActivation(new ActivationIdentity()))
                                            .addAgg(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(resNrofChannels))
                                            .andFinally(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(resNrofChannels))
                                            .addAgg(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(resNrofChannels))
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(resNrofChannels))
                                            .andFinally(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(resNrofChannels))
                                            .done()
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(2*resNrofChannels))
                                            .andFinally(new SeBlock())
                                            //.andFinally(new DropOut().setDropProb(dropOutProb))
                                            .andThenStack(2)
                                            .aggOf(new Dense())
                                            .andFinally(new DropOut().setDropProb(dropOutProb))
                                            .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                                                    .setAlpha(0.6)
                                                    .setLambda(lambda)));
                            modelData.add(new GenericModelHandle(
                                    trainIter,
                                    evalIter,
                                    new GraphModelAdapter(builder.buildGraph()),
                                    builder.name()));
                        })
                )
        );

        // Same thing as above but with factorized convolutions (does not seem to improve performance)
        IntStream.of(5).forEach(resDepth ->
                DoubleStream.of(0).forEach(dropOutProb ->
                        DoubleStream.of(0.004).forEach(lambda -> {
                            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                                    createStem(pool, lrSched, momSched, resNrofChannels)
                                            .andThenStack(resDepth)
                                            .res()
                                            .aggFork()
                                            .add(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(resNrofChannels)
                                                    .setActivation(new ActivationIdentity()))
                                            .addAgg(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(resNrofChannels))



                                            .andThen(new Conv2D()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize_h(3)
                                                    .setKernelSize_w(1)
                                                    .setActivation(new ActivationIdentity())
                                                    .setNrofKernels(resNrofChannels))
                                            .andFinally(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize_h(1)
                                                    .setKernelSize_w(3)
                                                    .setNrofKernels(resNrofChannels))


                                            .addAgg(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(resNrofChannels))
                                            .andThen(new Conv2D()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize_h(3)
                                                    .setKernelSize_w(1)
                                                    .setActivation(new ActivationIdentity())
                                                    .setNrofKernels(resNrofChannels))
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize_h(1)
                                                    .setKernelSize_w(3)
                                                    .setNrofKernels(resNrofChannels))
                                            .andThen(new Conv2D()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize_h(3)
                                                    .setKernelSize_w(1)
                                                    .setActivation(new ActivationIdentity())
                                                    .setNrofKernels(resNrofChannels))
                                            .andFinally(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize_h(1)
                                                    .setKernelSize_w(3)
                                                    .setNrofKernels(resNrofChannels))

                                            .done()
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(2*resNrofChannels * resblockOutFac))
                                            .andFinally(new SeBlock())
                                            //.andFinally(new DropOut().setDropProb(dropOutProb))
                                            .andThenStack(2)
                                            .aggOf(new Dense())
                                            .andFinally(new DropOut().setDropProb(dropOutProb))
                                            .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                                                    .setAlpha(0.6)
                                                    .setLambda(lambda)));
                            modelData.add(new GenericModelHandle(
                                    trainIter,
                                    evalIter,
                                    new GraphModelAdapter(builder.buildGraph()),
                                    builder.name()));
                        })
                )
        );
    }

    private BlockBuilder.RootBuilder createStem(LayerBlockConfig pool, ISchedule lrSched, ISchedule momSched, int resNrofChannels) {
        return new BlockBuilder()
                .setNamePrefix(namePrefix)
                // .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.EPOCH, 0.001, 10, 2)))
                .setUpdater(new Nesterovs(lrSched, momSched))
                .first(new ConvType(inputShape))
                .andThen(new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setKernelSize(3)
                        .setNrofKernels(64))
                .andThen(pool)
                .andThen(new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setKernelSize(3)
                        .setNrofKernels(128))
                .andThen(pool)
                .andThen(new SeBlock())
                .andThen(new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setKernelSize(3)
                        .setNrofKernels(2 * resNrofChannels))
                .andThen(pool)
                .andThen(new SeBlock());
    }
}
