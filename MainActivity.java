package com.example.sr.progressbar;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btn=(Button)findViewById(R.id.button);
        final EditText et=(EditText)findViewById(R.id.editText);
        final ProgressBar pg=(ProgressBar)findViewById(R.id.progressBar);
     btn.setOnClickListener(new View.OnClickListener() {
         @Override
         public void onClick(View view) {
             int sayi=Integer.parseInt(String.valueOf(et.getText()));
             pg.getMax(sayi);
         }
     });
    }
}
