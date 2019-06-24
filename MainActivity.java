package com.example.sr.uygulama1;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

public class MainActivity extends AppCompatActivity {
    TextView tv1,tv2;
    Button buton;
    Button buton2,buton7;
    SeekBar sb1;
    EditText et1;
    CheckBox cb1;
    CheckBox cb2;
    CheckBox cb3;
    CheckBox cb4;
    Button button3;
    ToggleButton tb1,tb2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //Elemanlaraa erişiyorum
        tv1=findViewById(R.id.textView);
        tv2=findViewById(R.id.textView2);
        et1=findViewById(R.id.et1);
        sb1=findViewById(R.id.sb1);
        cb1=findViewById(R.id.checkBox);
        cb2=findViewById(R.id.checkBox);
        cb3=findViewById(R.id.checkBox);
        cb4=findViewById(R.id.checkBox);
        tb1=(ToggleButton)findViewById(R.id.toggleButton);
        tb2=(ToggleButton)findViewById(R.id.toggleButton2);
        buton7=(Button)findViewById((R.id.button7));
        button3=(Button) findViewById(R.id.button3);
        buton7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                StringBuffer results=new StringBuffer();
                results.append("Toggle Buton 1"+tb1.getText());
                results.append("Toggle Buton 2"+tb2.getText());
                Toast.makeText(MainActivity.this,results.toString(),Toast.LENGTH_LONG).show();
            }
        });
        button3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent ekran1_gecis=new Intent();
                startActivity(ekran1_gecis);
            }
        });
        buton=(Button)findViewById(R.id.button);
        buton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //kodlar
                tv1.setText(et1.getText());

            }
        });
        buton2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                double toplam=000;
                if(cb1.isChecked())
                    toplam+=30;
                if(cb2.isChecked())
                    toplam+=50;
                if(cb3.isChecked())
                    toplam+=5;
                if(cb4.isChecked())
                    toplam+=1;
                Toast.makeText(MainActivity.this, ("Toplm tutar:"+toplam), Toast.LENGTH_LONG).show();
            }
        });
        sb1.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean b) {
                sb1.setMax(100);
                tv1.setTextSize(progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
    }
}
