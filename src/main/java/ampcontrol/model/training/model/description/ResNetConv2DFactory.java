package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

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

        final LayerBlockConfig zeroPad3x3 = new ZeroPad().setPad(1);
        final LayerBlockConfig pool = new Pool2D().setSize(3).setStride(3); final int resblockOutFac = 1;
       // final LayerBlockConfig pool = new MinMaxPool().setSize(3).setStride(3); final int resblockOutFac = 2;

        IntStream.of(5).forEach(resDepth ->
            DoubleStream.of(0).forEach(dropOutProb ->
                DoubleStream.of(0.003).forEach(lambda -> {
                    ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                            new BlockBuilder()
                            .setNamePrefix(namePrefix)
                           // .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.EPOCH, 0.001, 10, 2)))
                                    .setUpdater(new Nesterovs(new SawTooth(200, 1e-6,0.001)))
                            .first(new ConvType(inputShape))
                            .andThen(zeroPad3x3)
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize(3)
                                    .setNrofKernels(64))
                            .andThen(pool)
                            .andThen(zeroPad3x3)
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize(3)
                                    .setNrofKernels(128))
                            .andThen(pool)
                            .andThen(new SeBlock())
                            .andThen(zeroPad3x3)
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize(3)
                                    .setNrofKernels(128))
                            .andThen(pool)
                            .andThen(new SeBlock())
                            .andThenStack(resDepth)
                            .res()
                            .aggOf(new Conv2DBatchNormAfter()
                                    .setKernelSize(1)
                                    .setNrofKernels(64))
                            .andThen(zeroPad3x3)
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize(3)
                                    .setNrofKernels(128))
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize(1)
                                    .setNrofKernels(128 * resblockOutFac))
                            //.andThen(zeroPad3x3)
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
}
