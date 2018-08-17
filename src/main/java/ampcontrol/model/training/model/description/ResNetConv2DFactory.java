package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description of some homebrewed resnets with 2D convolutions
 *
 * @author Christian Skärby
 */
public class ResNetConv2DFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public ResNetConv2DFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
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

        final LayerBlockConfig pool = new Pool2D().setSize(3).setStride(3);
        final int resblockOutFac = 1;
        // final LayerBlockConfig pool = new MinMaxPool().setSize(3).setStride(3); final int resblockOutFac = 2;

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.05));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        IntStream.of(5).forEach(resDepth ->
                Stream.of(new IdBlock(), new SeBlock()).forEach(seOrId ->
                        DoubleStream.of(0.003).forEach(lambda -> {
                            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                                    new BlockBuilder()
                                            .setNamePrefix(namePrefix)
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
                                            .andThen(seOrId)
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(128))
                                            .andThen(pool)
                                            .andThen(seOrId)
                                            .andThenStack(resDepth)
                                            .res()
                                            .aggOf(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(64))
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setConvolutionMode(ConvolutionMode.Same)
                                                    .setKernelSize(3)
                                                    .setNrofKernels(128))
                                            .andThen(new Conv2DBatchNormAfter()
                                                    .setKernelSize(1)
                                                    .setNrofKernels(128 * resblockOutFac))
                                            //.andThen(zeroPad3x3)
                                            .andFinally(seOrId)
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
    }
}
