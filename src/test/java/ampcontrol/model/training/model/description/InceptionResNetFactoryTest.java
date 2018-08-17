package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link InceptionResNetFactory}
 *
 * @author Christian Skärby
 */
public class InceptionResNetFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer()
                .initialize(
                        ((tr, ev, is, np, md) -> new InceptionResNetFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
                );
    }
}