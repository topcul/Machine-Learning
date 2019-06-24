package com.example.sr.uygulama1;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;

/**
 * Created by SR on 28.2.2018.
 */

public class activity2 extends AppCompatActivity {
    Button anasayfa,ucuncusayfa;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity2);
        anasayfa=(Button)findViewById(R.id.button3);
        anasayfa=(Button)findViewById(R.id.button5);
        anasayfa.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

            }
        });
        //ucuncusayfa.setOnClickListener(new View.OnClickListener() {
      //      @Override
        //    public void onClick(View view) {
          //      Intent ekran3_gecis=new Intent(MainActivity.this,Activity3.class);
          //      startActivity();
          //  }
        //});

}
}
