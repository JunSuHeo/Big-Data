//Value for store latitude & longtitude
var addr = '';

// If start page, excute this function
$(document).ready(function(){
    // Request for init option value to flask
    $.get("/init", {str: "selectbox"}, function(res){
        var strSelect = "<option value=\"업종선택\">업종선택</option>";

        //Add option values
        for(var i = 0; i < res.length; i++){
            strSelect += "<option value=\"" + res[i] + "\">" + res[i] + "</option>";
        }
        document.getElementById("service").innerHTML = strSelect;
    });
});

// Init Google map
function initMap(){
    // First location
    var myLatlng = {lat: 37.549625, lng: 126.989123};

    // Set zoom & location
    var map = new google.maps.Map(
        document.getElementById('map'), {zoom: 13, center: myLatlng});
    
    // Set first content
    var infoWindow = new google.maps.InfoWindow({
            content: 'Lat/Lng', position: myLatlng
    });

    infoWindow.open(map);

    // Add click listener
    map.addListener('click', function(mapsMouseEvent){
        infoWindow.close();

        addr = mapsMouseEvent.latLng.toString();

        // Slicing latitude & longtitude 
        var addrSlice = addr.slice(1, -1);
        var addrSplit = addrSlice.split(', ', 2);
        var addrLat = addrSplit[0].slice(0, 9);
        var addrLog = addrSplit[1].slice(0, 9);

        // Set new window content
        infoWindow = new google.maps.InfoWindow({position: mapsMouseEvent.latLng});
        infoWindow.setContent("This is right?");
        infoWindow.open(map);

        // Set latitude & longtitude
        $("#lat").val(addrLat);
        $("#log").val(addrLog);

        // Request where this place's service code name
        $.get("/call", {addr: addr}, function(data){
            $("#guesses").val(data);
        });
    });
}

// button click action
function btn_click(){
    var result_lat = $("#lat").val();
    var result_log = $("#log").val();
    var result_guesses = $("#guesses").val();
    var result_service = $("#service").val();

    // If one of text is null, then alert "select again"
    if(result_lat == "" || result_log == "" || result_guesses == "" || result_service == "업종선택"){
        alert("다시선택");
    }
    // All of text is exist, start analysis service
    else{
        // Request start analysis service
        $.get("/service", {code_name: result_guesses, service_name: result_service}, function(res){
            // If data is lacked, display cannot analysis
            if(res == "분석 불가"){
                $("#guess_title").text("해당 상권에 데이터가 부족하거나 업종이 존재하지 않아 분석할 수 없습니다.");
                $("#guess_price").text("");
                $("#guess_value").text("");
            }
            // If analysis is success, display analysis information
            else{
                $("#guess_price").text("예상 월 매출 금액 : ₩" + res[0]);
                $("#guess_title").text("가장 점수가 높았던 3가지 변수 요인");
                $("#guess_value").text(res[1] + ", " + res[2] + ", " + res[3]);
            }
        });
    }
}