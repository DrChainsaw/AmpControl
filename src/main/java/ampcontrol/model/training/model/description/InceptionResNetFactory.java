package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.Scale;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description of some homebrewed inception resnets with 2D convolutions
 *
 * @author Christian Skärby
 */
public class InceptionResNetFactory {
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public InceptionResNetFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
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
        final ISchedule lrSched = new Mul(new Step(4000, new Exponential(0.2)),
                new SawTooth(schedPeriod, 1e-6, 0.1));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        final int resNrofChannels = 96 / 2;
        IntStream.of(5 ,10).forEach(resDepth ->
                Stream.of(new IdBlock(), new SeBlock()).forEach(seOrIdBlock ->
                        DoubleStream.of(0.002).forEach(lambda -> {
                            ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                                    createSmallStem(lrSched, momSched, resNrofChannels)
                                            .andThenStack(resDepth)
                                            .aggRes()
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
                                                    .setNrofKernels(2 * resNrofChannels))
                                            .andFinally(new Scale(0.1))
                                            .andFinally(seOrIdBlock)
                                            //.andFinally(new DropOut().setDropProb(dropOutProb))
                                            .andThenStack(2)
                                            .of(new Dense())
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

//        // Same thing as above but with factorized convolutions (does not seem to improve performance)
//        IntStream.of(5).forEach(resDepth ->
//                DoubleStream.of(0).forEach(dropOutProb ->
//                        DoubleStream.of(0.004).forEach(lambda -> {
//                            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
//                                    createStem(pool, lrSched, momSched, resNrofChannels)
//                                            .andThenStack(resDepth)
//                                            .res()
//                                            .aggFork()
//                                            .add(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(1)
//                                                    .setNrofKernels(resNrofChannels)
//                                                    .setActivation(new ActivationIdentity()))
//                                            .addAgg(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(1)
//                                                    .setNrofKernels(resNrofChannels))
//
//
//                                            .andThen(new Conv2D()
//                                                    .setConvolutionMode(ConvolutionMode.Same)
//                                                    .setKernelSize_h(3)
//                                                    .setKernelSize_w(1)
//                                                    .setActivation(new ActivationIdentity())
//                                                    .setNrofKernels(resNrofChannels))
//                                            .andFinally(new Conv2DBatchNormAfter()
//                                                    .setConvolutionMode(ConvolutionMode.Same)
//                                                    .setKernelSize_h(1)
//                                                    .setKernelSize_w(3)
//                                                    .setNrofKernels(resNrofChannels))
//
//
//                                            .addAgg(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(1)
//                                                    .setNrofKernels(resNrofChannels))
//                                            .andThen(new Conv2D()
//                                                    .setConvolutionMode(ConvolutionMode.Same)
//                                                    .setKernelSize_h(3)
//                                                    .setKernelSize_w(1)
//                                                    .setActivation(new ActivationIdentity())
//                                                    .setNrofKernels(resNrofChannels))
//                                            .andThen(new Conv2DBatchNormAfter()
//                                                    .setConvolutionMode(ConvolutionMode.Same)
//                                                    .setKernelSize_h(1)
//                                                    .setKernelSize_w(3)
//                                                    .setNrofKernels(resNrofChannels))
//                                            .andThen(new Conv2D()
//                                                    .setConvolutionMode(ConvolutionMode.Same)
//                                                    .setKernelSize_h(3)
//                                                    .setKernelSize_w(1)
//                                                    .setActivation(new ActivationIdentity())
//                                                    .setNrofKernels(resNrofChannels))
//                                            .andFinally(new Conv2DBatchNormAfter()
//                                                    .setConvolutionMode(ConvolutionMode.Same)
//                                                    .setKernelSize_h(1)
//                                                    .setKernelSize_w(3)
//                                                    .setNrofKernels(resNrofChannels))
//
//                                            .done()
//                                            .andThen(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(1)
//                                                    .setNrofKernels(2 * resNrofChannels * resblockOutFac))
//                                            .andFinally(new SeBlock())
//                                            //.andFinally(new DropOut().setDropProb(dropOutProb))
//                                            .andThenStack(2)
//                                            .aggOf(new Dense())
//                                            .andFinally(new DropOut().setDropProb(dropOutProb))
//                                            .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
//                                                    .setAlpha(0.6)
//                                                    .setLambda(lambda)));
//                            modelData.add(new GenericModelHandle(
//                                    trainIter,
//                                    evalIter,
//                                    new GraphModelAdapter(builder.buildGraph()),
//                                    builder.name()));
//                        })
//                )
//        );
    }

    private BlockBuilder.RootBuilder createSmallStem(ISchedule lrSched, ISchedule momSched, int resNrofChannels) {
        final LayerBlockConfig pool = new Pool2D().setSize(3).setStride(3);

        return new BlockBuilder()
                .setNamePrefix(namePrefix)
                .setUpdater(new Nesterovs(lrSched, momSched))
                .first(new ConvType(inputShape))
                .andThen(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(64))
                .andThen(pool)
               .andThen(new Conv2DBatchNormAfter()
               .setKernelSize(3)
               .setNrofKernels(128))
                .andThen(pool)
                .andThen(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                .setNrofKernels(2*resNrofChannels))
                ;
    }
}
