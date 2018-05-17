package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link SampleCnn2DFactory}
 *
 * @author Christian Skärby
 */
public class SampleCnn2DFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer()
                .setInputSize(new int[] {1, 4410, 2})
                .initialize(
                ((tr, ev, is, np, md) -> new SampleCnn2DFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
        );
    }
}