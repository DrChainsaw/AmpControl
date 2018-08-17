package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.GlobMeanMax;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

/**
 * Description of a bunch of architectures which belong to a family of stacked 2D convolutional neural networks.
 *
 * @author Christian Skärby
 */
public class StackedConv2DFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public StackedConv2DFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
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
//            final LayerBlockConfig zeroPad4x4 = new ZeroPad()
//                    .setPad_h_top(1)
//                    .setPad_h_bot(2)
//                    .setPad_w_left(1)
//                    .setPad_w_right(2);

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.01));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        Stream.of(new Pool2D().setSize(2).setStride(2)).forEach(pool -> {
            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                    new BlockBuilder()
                            .setUpdater(new Nesterovs(lrSched, momSched))
                            .setNamePrefix(namePrefix)
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
                                    .setNrofKernels(256))
                            .andThen(pool)

                            .andThen(new GlobMeanMax())
                            .andThenStack(2)
                            .of(new Dense()
                                    .setHiddenWidth(512)
                                    .setActivation(new ActivationReLU()))
                            .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                                    .setAlpha(0.6)
                                    .setLambda(0.004)));
            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(builder.buildGraph()),
                    builder.name()));
        });

        // Current best score with lgsc 96.0. Also performs very well in practice
        //final LayerBlockConfig pool = new MinMaxPool().setSize(2).setStride(2);
//        final LayerBlockConfig pool = new Pool2D().setSize(2).setStride(2);
//        Stream.of(new IdBlock(), new SeBlock()).forEach(afterConvBlock ->
//                IntStream.of(3).forEach(kernelSize ->
//                        DoubleStream.of(0).forEach(dropOutProb -> {
//                            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
//                                    new BlockBuilder()
//                                            .setUpdater(new Nesterovs(lrSched, momSched))
//                                            .setNamePrefix(namePrefix)
//                                            .first(new ConvType(inputShape))
//                                            .andThenStack(2)
//                                            .of(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(kernelSize)
//                                                    .setNrofKernels(64))
//                                            .andThen(pool)
//
//                                            .andThenStack(2)
//                                            .of(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(kernelSize)
//                                                    .setNrofKernels(128))
//                                            .andThen(afterConvBlock)
//                                            .andThen(pool)
//
//                                            .andThenStack(2)
//                                            .of(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(kernelSize)
//                                                    .setNrofKernels(256))
//                                            .andThen(afterConvBlock)
//                                            .andThen(pool)
//
//                                            .andThenStack(2)
//                                            .of(new Conv2DBatchNormAfter()
//                                                    .setKernelSize(kernelSize)
//                                                    .setNrofKernels(512))
//                                            .andThen(pool)
//
//                                            //.andThen(new GlobMeanMax())
//                                            .andThenStack(2)
//                                            .aggOf(new Dense()
//                                                    .setHiddenWidth(512)
//                                                    .setActivation(new ActivationReLU()))
//                                            .andFinally(new DropOut().setDropProb(dropOutProb))
//                                            .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
//                                                    .setAlpha(0.6)
//                                                    .setLambda(0.004)));
//                            modelData.add(new GenericModelHandle(
//                                    trainIter,
//                                    evalIter,
//                                    new GraphModelAdapter(builder.buildGraph()),
//                                    builder.name()));
//                        })
//                )
//        );
    }
}
